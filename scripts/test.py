import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

from src.utils import *
from src.network import ColumnNetwork




### SETTING UP ###

nr_inputs = 1
device = 'cpu'

col_params = load_config('../config/model_params.toml')

network_input = {'nr_areas': 4,
                 'areas': ['v1', 'v2', 'v4', 'pitv'],
                 'nr_columns_per_area': [1, 1, 1, 1],
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



# ### RUN THE NETWORK ###
#
# network.stim = torch.tensor([40.])
#
# ode_output = odeint(network, initial_state, time_vec)
#
# mem_adap_split = ode_output.shape[-1] // 2
# firing_rates = compute_firing_rate(ode_output[:, :, :mem_adap_split] - ode_output[:, :, mem_adap_split:])
# firing_rates = firing_rates.squeeze(1)
#
# firing_rates = firing_rates.detach().numpy()
# for i in range(firing_rates.shape[-1]):
#     print(i)
#     plt.plot(firing_rates[:, i])
#     plt.show()



### FUNCTIONS FOR TRAINING

def mask_weights(network):
    network.areas['0'].input_weights.grad *= network.areas['0'].input_mask  # input weights

    for area_idx in range(1, network.nr_areas):  # feedforward weights, skip first area
        network.areas[str(area_idx)].feedforward_weights.grad *= network.areas[str(area_idx)].feedforward_mask

    # for area_idx in range(0, network.nr_areas - 1):  # feedback weights, skip last area
    #     network.areas[str(area_idx)].feedback_weights.grad *= network.areas[str(area_idx)].feedback_mask

def visualize_weights(network, train_iter):
    if not os.path.exists('../results/png'):
        os.makedirs('../results/png')

    for name, param in network.named_parameters():
        param_data = param.detach().cpu().numpy()

        if np.sum(param_data) != 0:
            fig, ax = plt.subplots(figsize=(13, 7))

            if param_data.ndim == 2:  # 2D weight matrices: use heatmap
                heatmap = ax.imshow(param_data, cmap="viridis", interpolation="nearest")
                fig.colorbar(heatmap, ax=ax)
                ax.set_title(f"Weight Matrix: {name}")

            # Clean filename (remove problematic characters)
            clean_name = name.replace('.', '_')
            plt.savefig('../results/png/{}_{:02d}'.format(clean_name, train_iter + 1))
            plt.close(fig)



### TRAIN THE NETWORK ###

num_iterations = 1000
optimizer = torch.optim.Adam(network.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08)

for iter in range(num_iterations):
    optimizer.zero_grad()

    stims = torch.tensor([10., 12., 14., 16., 18.,
                          20., 22., 24., 26., 28.,
                          30., 32., 34., 36., 38.,
                          40.])
    batch_output = torch.zeros_like(stims)

    for i, stim in enumerate(stims):

        network.stim = stim.unsqueeze(0)

        ode_output = odeint(network, initial_state, time_vec)

        mem_adap_split = ode_output.shape[-1] // 2
        firing_rates = compute_firing_rate(ode_output[:, :, :mem_adap_split] - ode_output[:, :, mem_adap_split:])
        firing_rates = firing_rates.squeeze(1)

        output_layer_fr = firing_rates[:, -4]  # layer 5 of last column
        batch_output[i] = torch.mean(output_layer_fr[-300:], dim=0)  # average over last 300 time steps

        # print(torch.mean(output_layer_fr[-300:], dim=0))
        # plt.plot(output_layer_fr.detach().numpy())
        # plt.show()

    # Calculate loss and backprop
    loss = torch.mean(abs(stims - batch_output))
    loss.backward()

    print('Iter {:02d} | Total Loss {:.5f}'.format(iter + 1, loss.item()))

    mask_weights(network)
    optimizer.step()

    for name, param in network.named_parameters():
        param.data.clamp_(min=0.0)  # synapse weights can not be negative

    # Every n batches, print results, visualize weights and save the current network
    with torch.no_grad():
        if iter % 5 == 0:
            print('L5e firing rates {}'.format(batch_output))
            visualize_weights(network, iter)
            with open('../models/network_trained_synapses.pkl', 'wb') as f:
                pickle.dump(network, f)


