import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from src.network import ColumnNetwork
from src.utils import *




def visualize_weights(network, train_iter):
    # if not os.path.exists('../results/png'):
    #     os.makedirs('../results/png')

    for name, param in network.named_parameters():
        param_data = param.detach().cpu().numpy()

        if np.sum(param_data) != 0:
            fig, ax = plt.subplots(figsize=(13, 7))

            if param_data.ndim == 2:  # 2D weight matrices: use heatmap
                heatmap = ax.imshow(param_data, cmap="viridis", interpolation="nearest")
                fig.colorbar(heatmap, ax=ax)
                ax.set_title(f"Weight Matrix: {name}")
            elif param_data.ndim == 1:  # output weights are 1D: use bar plot
                ax.bar(np.arange(len(param_data)), param_data, color="slateblue")
                ax.set_title(f"Weight Vector: {name}")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")

            plt.show()

            # Clean filename (remove problematic characters)
            clean_name = name.replace('.', '_')
            # plt.savefig('../results/png/{}_{:02d}'.format(clean_name, train_iter + 1))
            # plt.close(fig)

def prepare_ds(digits_to_include):

    # Load dataset
    digits = datasets.load_digits()

    # Images and labels
    X = digits.images  # shape: (n_samples, 8, 8)
    y = digits.target

    # Only data instances with a label in digits_to_include
    mask = np.isin(y, digits_to_include)
    X = X[mask]
    y = y[mask]

    # Flatten the images
    n_samples = len(X)
    X = X.reshape((n_samples, -1))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=True
    )

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # # One hot encode the labels
    # y_train = nn.functional.one_hot(y_train, len(digits_to_include)).to(torch.float32)
    # y_test = nn.functional.one_hot(y_test, len(digits_to_include)).to(torch.float32)

    return X_train, X_test, y_train, y_test

def init_network(nr_inputs, device):
    col_params = load_config('../config/model_params.toml')

    # network_input = {'nr_areas': 4,
    #                  'areas': ['v1', 'v2', 'v4', 'pitv'],
    #                  'nr_columns_per_area': [144, 64, 16, 8],
    #                  'nr_input_units': nr_inputs}
    # network_input = {'nr_areas': 4,
    #                  'areas': ['v1', 'v2', 'v4', 'pitv'],
    #                  'nr_columns_per_area': [72, 32, 8, 4],
    #                  'nr_input_units': nr_inputs}
    network_input = {'nr_areas': 4,
                     'areas': ['v1', 'v2', 'v4', 'pitv'],
                     'nr_columns_per_area': [36, 16, 4, 4],
                     'nr_input_units': nr_inputs}
    network = ColumnNetwork(col_params, network_input, device)
    num_columns = sum(network_input['nr_columns_per_area'])

    stim_duration = 0.5
    dt = 1e-3
    time_steps = int(stim_duration * 2 / dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    initial_state = torch.zeros(num_columns * 8 * 2)  # 2 state variables
    initial_state = initial_state.unsqueeze(0)

    network = network.to(device).to(torch.float32)
    initial_state = initial_state.to(device).to(torch.float32)
    time_vec = time_vec.to(device).to(torch.float32)

    network.time_vec = time_vec
    return network, time_vec, initial_state

def train_digit_classification(device, batch_size=16, nr_epochs=50):

    # Get train and test set
    digits_to_include = [0, 1, 2, 3]
    X_train, X_test, y_train, y_test = prepare_ds(digits_to_include)
    nr_inputs = X_train.shape[1]

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    # Initialize the network and associated variables
    network, time_vec, initial_state = init_network(nr_inputs, device)
    # visualize_weights(network, 0)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    # Training loop
    for epoch in range(nr_epochs):

        for stims, labels in train_loader:
            model_predictions = torch.zeros(batch_size, len(digits_to_include))

            for stim_idx, stim in enumerate(stims):

                stim_as_image = stim.reshape((8, 8))

                # Set image as stimulus
                network.stim = stim

                # Run the network and compute the firing rates
                ode_output = odeint(network, initial_state, time_vec)
                mem_adap_split = ode_output.shape[-1] // 2
                firing_rates = compute_firing_rate(ode_output[:, :, :mem_adap_split] - ode_output[:, :, mem_adap_split:])
                # # Plot firing rates
                # firing_rates = firing_rates.squeeze(1).detach().numpy()
                # col_idx = 0 # (288+128)*2
                # for i in range(col_idx, firing_rates.shape[-1]):
                #     print(i - col_idx)
                #     plt.plot(firing_rates[:, i])
                #     plt.show()

                # Get the firing rates from the final area columns
                size_last_area = network.nr_columns_per_area[-1]
                num_pops_last_area = size_last_area * 8  # 8 populations

                firing_rates_last_area = firing_rates[-1, 0, -num_pops_last_area:]
                firing_rates_last_area_L5 = firing_rates_last_area * network.output_weights
                separate_final_columns = torch.tensor_split(firing_rates_last_area_L5, size_last_area)

                for column_idx, column in enumerate(separate_final_columns):
                    model_predictions[stim_idx, column_idx] = torch.sum(column)

            # Compute loss and backprop
            loss = criterion(model_predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate



if __name__ == '__main__':
    device = 'cpu'

    train_digit_classification(device)
