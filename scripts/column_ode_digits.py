import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pickle
import random
import time
from datetime import datetime

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from src.network import ColumnNetwork
from src.utils import *





def set_seed(seed):
    '''
    Setting the random seed for reproducability.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_results_folder():
    '''
    Make a folder in ../results to store the training results.
    '''
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_path = f'../results/results_{timestamp}'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return results_path

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

def heatmap_model_output(model_preds):

    y_true = model_preds[:, -1:]
    y_pred = np.argmax(model_preds[:, :-1], axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    ConfusionMatrixDisplay(cm).plot(cmap="magma")
    plt.title("Confusion Matrix")
    plt.show()

    # Activations heatmap
    logits = model_preds[:, :-1]
    labels = model_preds[:, -1]

    # sort by label
    sorted_idx = labels.argsort()
    logits_sorted = logits[sorted_idx]

    plt.figure(figsize=(5, 10))
    sns.heatmap(logits_sorted.numpy(), cmap="magma", vmax=20.0)
    plt.xlabel("Class")
    plt.ylabel("Sample (sorted by label)")
    plt.show()

def run_batch(network, time_vec, initial_state, model_predictions, stims, device):
    '''
    Runs a batch of images through the network. Returns the model predictions
    (i.e. the final firing rates of the output columns) and the raw firing rates.
    '''
    # Set image as stimulus
    stims = stims.to(device)
    network.set_stim(stims)

    # Run the network and compute the firing rates
    ode_output = odeint(network, initial_state, time_vec).to(device)
    mem_adap_split = ode_output.shape[-1] // 2
    firing_rates = compute_firing_rate(ode_output[:, :, :mem_adap_split] - ode_output[:, :, mem_adap_split:])

    # # Plot firing rates
    # firing_rates_plot = firing_rates.detach().cpu().numpy()
    # col_idx = 1024
    # for i in range(len(stims)):
    #     print(i)
    #     for j in range(col_idx, firing_rates_plot.shape[-1] - 8):
    #         print(j - col_idx)
    #         plt.plot(firing_rates_plot[:, i, j])
    #         plt.plot(firing_rates_plot[:, i, j+8])
    #         plt.show()

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

def train_digit_classification(digits_to_include,
                               device,
                               batch_size=16,
                               nr_epochs=50,
                               lr=1e-2,
                               lambda_suppression=1e-1,
                               lambda_magnitude=1e-2,
                               lambda_ei=1e+0):
    '''
    Train a Column Network to classify handwritten digits.
    '''
    # Make results folder
    results_path = make_results_folder()

    # Get train and test set
    X_train, X_test, y_train, y_test = prepare_ds(digits_to_include, padding=1)
    nr_inputs = X_train.shape[1]

    # for i, stim in enumerate(X_train):  # take a peek at the images
    #     print(y_train[i].item())
    #     stim = np.array(stim).reshape((10,10))
    #     plt.imshow(stim, cmap=plt.cm.gray_r, interpolation="nearest")
    #     plt.show()

    # DataLoader for train set
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize the network and associated variables
    network, time_vec, initial_state = init_network(nr_inputs, len(digits_to_include), batch_size, device)

    # # Load in existing network
    # network = load_pkl_file('../results/10_digits!/network_post_training_epoch_00.pkl')

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    # Training loop
    for epoch in range(0, nr_epochs):
        print('Epoch {}'.format(epoch))

        for stims, labels in train_loader:
            start = time.time()
            optimizer.zero_grad()
            network.constrain_weights()

            # Run the network with the training batch as input
            initial_train = torch.tile(initial_state, (batch_size, 1))
            model_predictions = torch.zeros(batch_size, len(digits_to_include))
            model_predictions, _ = run_batch(network, time_vec, initial_train, model_predictions, stims, device)

            # Compute loss and backprop
            unique_labels, mapped_labels = torch.unique(labels, return_inverse=True)
            ce_loss = criterion(model_predictions, mapped_labels)

            one_hot_labels = nn.functional.one_hot(mapped_labels, num_classes=len(digits_to_include))
            suppression = ((1 - one_hot_labels) * model_predictions).mean()
            L2_reg = (network.areas['0'].input_weights ** 2).mean()

            # >>> Compute E/I ratio
            W_in = network.areas['0'].input_weights
            num_columns = W_in.shape[0] // 8
            W_in_reshaped = W_in.view(num_columns, 8, -1)

            W_E = W_in_reshaped[:, 2, :]  # (columns, source)
            W_I = W_in_reshaped[:, 3, :]

            E_drive = W_E.abs().sum(dim=1)
            I_drive = W_I.abs().sum(dim=1)

            ratio = E_drive / (E_drive + I_drive + 1e-6)
            # ei_loss = (ratio - 0.61).sum()
            ei_loss = (ratio - 0.61).pow(2).sum()
            # <<<

            loss = ce_loss + (lambda_suppression * suppression) + (lambda_magnitude * L2_reg) + (lambda_ei * ei_loss)

            loss.backward()
            optimizer.step()

            print('Train loss | {:.5f} | {:.1f}s'.format(loss.item(), time.time() - start))

        # Evaluate with test set, after every epoch
        with torch.no_grad():
            network.constrain_weights()

            print('==================== TESTING ====================')
            initial_test = torch.tile(initial_state, (X_test.shape[0], 1))
            model_predictions = torch.zeros(X_test.shape[0], len(digits_to_include))
            model_predictions, firing_rates = run_batch(network, time_vec, initial_test, model_predictions, X_test, device)

            _, mapped_test_labels = torch.unique(y_test, return_inverse=True)
            test_loss = criterion(model_predictions, mapped_test_labels)
            print('Test loss CE {:.5f}'.format(test_loss.item()))

            one_hot_labels = nn.functional.one_hot(mapped_test_labels, num_classes=len(digits_to_include))
            suppression = ((1 - one_hot_labels) * model_predictions).mean()
            loss_supp = lambda_suppression * suppression
            print('Suppression {:.5f}'.format(loss_supp.item()))

            L2_reg = (network.areas['0'].input_weights ** 2).mean()
            print('L2 regularization {:.5f}'.format(lambda_magnitude * L2_reg.item()))

            # print('E/I ratio {:.5f}'.format(lambda_ei * ei_loss))

            test_mae = torch.mean(abs(model_predictions - (one_hot_labels * 20.0)))
            print('Test loss MAE {:.5f}'.format(test_mae.item()))

            test_acc = (y_test == torch.argmax(model_predictions, dim=1)).float().mean()
            print('Test accuracy {:.2f}'.format(test_acc))

            print('Model predictions and true labels')
            # heatmap_model_output(torch.concat((model_predictions, y_test.unsqueeze(1)), dim=-1))
            # print(torch.concat((model_predictions, y_test.unsqueeze(1)), dim=-1))

            # Save current network
            save_pkl_file('{}/network_post_training_epoch_{:02d}.pkl'.format(results_path, epoch), network)





if __name__ == '__main__':

    seed = 1
    set_seed(seed)

    digits_to_include = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    device = torch.device('mps')

    train_digit_classification(digits_to_include, device, batch_size=64, nr_epochs=200)

