import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import random
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from src.network import ColumnNetwork
from src.utils import *




def visualize_feature_maps_and_weights(network, firing_rates, labels, epoch):
    '''
    Visualize weights and feature maps during training.
    '''
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    # Weights
    for name, param in network.named_parameters():
        if param.requires_grad:
            fig, ax = plt.subplots(figsize=(18, 5))

            param_data = param.detach().cpu().numpy()
            heatmap = ax.imshow(param_data, cmap="magma", interpolation="nearest")
            fig.colorbar(heatmap, ax=ax)
            ax.set_title(f"Weight Matrix: {name}")

            clean_name = name.replace('.', '_')
            plt.savefig('../results/png/{}_epoch_{:02d}'.format(clean_name, epoch))
            plt.close(fig)

    # # Feature maps
    # for stim_idx in range(firing_rates.shape[0]):
    #     firing_rates_v1 = firing_rates[stim_idx, 500:, :576]
    #     firing_rates_v1_mean = torch.mean(firing_rates_v1, dim=0)
    #
    #     input_current_v2 = network.areas['1'].feedforward_weights * firing_rates_v1_mean
    #     column_0 = torch.sum(input_current_v2[:8, :], dim=0)
    #     column_1 = torch.sum(input_current_v2[8:, :], dim=0)
    #
    #     def sum_over_source_columns(x):
    #         x_grouped = x.view(-1, 8)
    #         return torch.sum(x_grouped, dim=1)
    #
    #     column_0 = sum_over_source_columns(column_0)
    #     column_1 = sum_over_source_columns(column_1)
    #
    #     column_0_horizontal = column_0[::2].reshape((6, 6))
    #     column_0_vertical = column_0[1::2].reshape((6, 6))
    #
    #     column_1_horizontal = column_1[::2].reshape((6, 6))
    #     column_1_vertical = column_1[1::2].reshape((6, 6))
    #
    #     feature_maps = {
    #         "col_0_horizontal": column_0_horizontal,
    #         "col_0_vertical": column_0_vertical,
    #         "col_1_horizontal": column_1_horizontal,
    #         "col_1_vertical": column_1_vertical,
    #     }
    #
    #     for name, fmap in feature_maps.items():
    #         heatmap = plt.imshow(fmap.detach().numpy(), cmap="magma", interpolation="nearest", vmin=0.0, vmax=50.0)  # inferno, magma
    #         plt.colorbar(heatmap)
    #         plt.savefig('../results/png/epoch_{:02d}_label_{:1d}_{}_{:02d}'.format(epoch, labels[stim_idx], name, stim_idx))
    #         plt.close()

def prepare_ds(digits_to_include, padding):
    '''
    Prepare the sklearn digit dataset by padding the images, flattening
    images to vectors and splitting the data into train and test sets.
    '''
    # Load dataset
    digits = datasets.load_digits()
    X = digits.images  # shape: (n_samples, 8, 8)
    y = digits.target

    # Pad the images
    if padding > 0:
        X = np.pad(X, ((0,0), (padding,padding), (padding,padding)))

    # Only data instances with a label in digits_to_include
    mask = np.isin(y, digits_to_include)
    X = X[mask]
    y = y[mask]

    # Flatten the images
    n_samples = len(X)
    X = X.reshape((n_samples, -1))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test

def mask_weights(network):
    '''
    Mask all the trainable parameters to make sure no illegal updates
    can be made.
    '''

    network.areas['0'].input_weights.grad *= network.areas['0'].input_mask

    for area_idx in range(1, network.nr_areas):  # feedforward weights, skip first area
        network.areas[str(area_idx)].feedforward_weights.grad *= network.areas[str(area_idx)].feedforward_mask

    # for area_idx in range(network.nr_areas):  # lateral weights
    #     network.areas[str(area_idx)].lateral_weights.grad *= network.areas[str(area_idx)].lateral_mask

def init_network(nr_inputs, nr_outputs, batch_size, device):
    '''
    Initialize the network, the time vector and the initial state that
    every network simulation should start from.
    '''
    col_params = load_config('../config/model_params.toml')

    network_input = {'nr_areas': 2,
                     'areas': ['v1', 'v2'],
                     'nr_columns_per_area': [128, nr_outputs],
                     'nr_input_units': nr_inputs}
    network = ColumnNetwork(col_params, network_input)
    num_columns = sum(network_input['nr_columns_per_area'])

    stim_duration = 0.5
    dt = 1e-3
    time_steps = int(stim_duration * 2 / dt)
    time_vec = torch.linspace(0., time_steps * dt, time_steps)

    initial_state = torch.zeros(1, num_columns * 8 * 2)  # 2 state variables

    network.time_vec = time_vec
    return network.to(device), time_vec.to(device), initial_state.to(device)

def run_batch(network, time_vec, initial_state, model_predictions, stims, device):
    '''
    Runs a batch of images through the network. Returns the model predictions
    (i.e. the final firing rates of the output columns) and the raw firing rates.
    '''
    # stim = torch.tensor([[0., 0., 0., 0., 8., 16., 0., 0.],
    #                      [0., 5., 13., 16., 16., 16., 0., 0.],
    #                      [0., 11., 16., 15., 12., 16., 0., 0.],
    #                      [0., 3., 8., 0., 8., 16., 0., 0.],
    #                      [0., 0., 0., 0., 8., 16., 3., 0.],
    #                      [0., 0., 0., 0., 8., 16., 4., 0.],
    #                      [0., 0., 0., 0., 7., 16., 7., 0.],
    #                      [0., 0., 0., 0., 10., 16., 8., 0.]]).reshape(64)
    # stim = torch.tensor([[0., 0., 4., 13., 14., 8., 0., 0.],
    #                      [0., 3., 14., 3., 1., 16., 3., 0.],
    #                      [0., 7., 9., 0., 0., 14., 6., 0.],
    #                      [0., 8., 4., 0., 0., 16., 4., 0.],
    #                      [0., 8., 6., 0., 0., 16., 0., 0.],
    #                      [0., 3., 11., 0., 1., 14., 0., 0.],
    #                      [0., 0., 12., 4., 6., 11., 0., 0.],
    #                      [0., 0., 5., 16., 14., 1., 0., 0.]]).reshape(64)
    # stim = torch.tensor([[0., 10., 0., 0., 8., 16., 0., 0.],
    #                      [16., 16., 16., 16., 16., 16., 0., 0.],
    #                      [0., 10., 0., 15., 12., 16., 0., 0.],
    #                      [0., 3., 8., 0., 8., 16., 0., 0.],
    #                      [0., 0., 0., 0., 8., 16., 3., 0.],
    #                      [0., 0., 0., 0., 8., 16., 4., 0.],
    #                      [0., 0., 0., 0., 7., 16., 7., 0.],
    #                      [0., 0., 0., 0., 10., 16., 8., 0.]]).reshape(64)
    # stim_as_image = stim.reshape((10, 10))

    # Set image as stimulus
    stims = stims.to(device)
    network.set_stim(stims)

    # Run the network and compute the firing rates
    ode_output = odeint(network, initial_state, time_vec).to(device)
    mem_adap_split = ode_output.shape[-1] // 2
    firing_rates = compute_firing_rate(ode_output[:, :, :mem_adap_split] - ode_output[:, :, mem_adap_split:])
    # # Plot firing rates
    # firing_rates_plot = firing_rates.squeeze(1).detach().numpy()
    # col_idx = 1024 # 576
    # for i in range(col_idx, firing_rates_plot.shape[-1] - 8):
    #     print(i - col_idx)
    #     plt.plot(firing_rates_plot[:, i])
    #     plt.plot(firing_rates_plot[:, i+8])
    #     plt.show()

    # Get the firing rates from the final area columns
    size_last_area = network.nr_columns_per_area[-1]
    num_pops_last_area = size_last_area * 8  # 8 populations

    fr_last_area = firing_rates[-300:, :, -num_pops_last_area:]
    fr_last_area = torch.mean(fr_last_area, dim=0)
    fr_output = fr_last_area * network.output_weights

    for col_idx in range(size_last_area):
        col = fr_output[:, col_idx * 8 : (col_idx+1) * 8]
        col_summed = torch.sum(col, dim=1)
        model_predictions[:, col_idx] = col_summed

    return model_predictions, firing_rates

def set_seed(seed):
    '''
    Setting the random seed for reproducability.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_digit_classification(device,
                               batch_size=16,
                               nr_epochs=50):
    '''
    Train a Column Network to classify handwritten digits.
    '''

    # Get train and test set
    digits_to_include = [0, 1]
    X_train, X_test, y_train, y_test = prepare_ds(digits_to_include, padding=1)
    nr_inputs = X_train.shape[1]

    # DataLoader for train set
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize the network and associated variables
    network, time_vec, initial_state = init_network(nr_inputs, len(digits_to_include), batch_size, device)

    # # Load in existing network
    # network = load_pkl_file('../results/__.pkl')

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-2)

    # Training loop
    for epoch in range(0, nr_epochs):
        print('Epoch {}'.format(epoch))

        for stims, labels in train_loader:
            start = time.time()

            # Run the network with the training batch as input
            initial_train = torch.tile(initial_state, (batch_size, 1))
            model_predictions = torch.zeros(batch_size, len(digits_to_include))
            model_predictions, _ = run_batch(network, time_vec, initial_train, model_predictions, stims, device)

            # Compute loss and backprop
            ce_loss = criterion(model_predictions, labels)
            one_hot_labels = nn.functional.one_hot(labels, num_classes=len(digits_to_include))
            mae = torch.mean(abs(model_predictions - (one_hot_labels * 20.0)))
            suppression = ((1 - one_hot_labels) * model_predictions).mean()
            loss = ce_loss + (0.1 * suppression)
            optimizer.zero_grad()
            loss.backward()
            mask_weights(network)  # no illegal updates
            optimizer.step()

            # Clamp the weights to ensure the weights are not below zero after updating (or are not higher than zero)
            for name, param in network.named_parameters():
                param.data.clamp_(min=0.0)  # weights can not be negative

            print('Train loss | {:.5f} | {:.1f}s'.format(loss.item(), time.time() - start))

        # Evaluate with test set, after every epoch
        with torch.no_grad():
            print('==================== TESTING ====================')
            initial_test = torch.tile(initial_state, (X_test.shape[0], 1))
            model_predictions = torch.zeros(X_test.shape[0], len(digits_to_include))
            model_predictions, firing_rates = run_batch(network, time_vec, initial_test, model_predictions, X_test, device)

            test_loss = criterion(model_predictions, y_test)
            print('Test loss CE {:.5f}'.format(test_loss.item()))

            one_hot_labels = nn.functional.one_hot(y_test, num_classes=len(digits_to_include))
            suppression = ((1 - one_hot_labels) * model_predictions).mean()
            loss_supp = test_loss + (0.1 * suppression)
            print('Test loss CE with suppression {:.5f}'.format(loss_supp.item()))

            test_mae = torch.mean(abs(model_predictions - (one_hot_labels * 20.0)))
            print('Test loss MAE {:.5f}'.format(test_mae.item()))

            test_acc = (y_test == torch.argmax(model_predictions, dim=1)).float().mean()
            print('Test accuracy {:.2f}'.format(test_acc))

            print('Model predictions and true labels')
            print(torch.concat((model_predictions, y_test.unsqueeze(1)), dim=-1))

            # Visualize results and save current network
            visualize_feature_maps_and_weights(network, firing_rates, y_test, epoch)
            save_pkl_file('../results/png/network_post_training_epoch_{:02d}.pkl'.format(epoch), network)





if __name__ == '__main__':
    device = torch.device('mps')
    seed = 1

    set_seed(seed)
    train_digit_classification(device)

'''
2 digits
PCA filters
padding=1
no lateral weights

cross-entropy loss with suppression=0.1
'''

