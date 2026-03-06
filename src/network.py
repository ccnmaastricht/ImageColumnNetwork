import numpy as np
import torch
from scipy.linalg import block_diag
import torch.nn as nn

from src.utils import *



class ColumnArea(torch.nn.Module):
    '''
    Initializes an area of columns along with all the relevant parameters
    to simulate their state activity over time.
    '''

    def __init__(self, column_parameters, area, num_columns):
        super().__init__()

        self.num_columns = num_columns
        self.area = area.lower()

        self._intialize_basic_parameters(column_parameters)
        self._initilize_population_parameters(column_parameters)
        self._initialize_connection_probabilities(column_parameters)
        self._initialize_synapses(column_parameters)

        self._build_all_weights()

    def _intialize_basic_parameters(self, column_parameters):
        """
        Initialize basic parameters for the columns.
        """
        # Basic parameters
        self.register_buffer("background_drive", torch.tensor(column_parameters['background_drive'], dtype=torch.float32)) # device
        self.register_buffer("adaptation_strength", torch.tensor(column_parameters['adaptation_strength'], dtype=torch.float32))  # device

        # Time constants and membrane resistance
        self.time_constants = column_parameters['time_constants']
        self.register_buffer("synapse_time_constant", torch.tensor(self.time_constants['synapse'], dtype=torch.float32)) # device
        self.register_buffer("membrane_time_constant", torch.tensor(self.time_constants['membrane'], dtype=torch.float32))  # device
        self.register_buffer("adapt_time_constant", torch.tensor(self.time_constants['adaptation'], dtype=torch.float32))  # device
        resistance = self.time_constants['membrane'] / column_parameters['capacitance']
        self.register_buffer("resistance", torch.tensor(resistance, dtype=torch.float32))  # device

    def _initilize_population_parameters(self, column_parameters):
        """
        Initialize the population sizes for the columns.
        """
        self.population_sizes = np.array(
            column_parameters['population_size'][self.area])
        self.population_sizes = np.tile(self.population_sizes, self.num_columns)

        self.num_populations = len(self.population_sizes)
        self.adaptation_strength = torch.tile(self.adaptation_strength, (self.num_columns,))

        self._make_in_ex_masks(self.num_columns)

    def _initialize_connection_probabilities(self, column_parameters):
        """
        Initialize the connection probabilities for the columns.
        """
        self.internal_connection_probabilities = torch.tensor(
            column_parameters['connection_probabilities']['internal'])

        # Copy internal connections n times along diagonal for n columns
        blocks = [self.internal_connection_probabilities] * self.num_columns
        self.connection_probabilities = block_diag(*blocks)

    def _initialize_synapses(self, column_parameters):
        """
        Initialize the synapse counts and synaptic strengths for the columns.
        """
        self.background_synapse_counts = torch.tensor(
            column_parameters['background_synapse_counts'][self.area])

        self.background_synapse_counts = torch.tile(
            self.background_synapse_counts, (self.num_columns,))

        self.baseline_synaptic_strength = column_parameters[
            'synaptic_strength']['baseline']

        self._compute_recurrent_synapse_counts()
        self._build_recurrent_synaptic_strength_matrix()

    def _compute_recurrent_synapse_counts(self):
        """
        Compute the number of synapses for recurrent connections based on the
        connection probabilities and population sizes.
        """
        log_numerator = np.log(1 - np.array(self.connection_probabilities))
        log_denominator = np.log(1 - 1 / np.array(np.outer(self.population_sizes, self.population_sizes)))

        recurrent_synapse_counts = log_numerator / log_denominator / self.population_sizes[:, None]
        self.recurrent_synapse_counts = torch.tensor(recurrent_synapse_counts, dtype=torch.float32)

    def _build_recurrent_synaptic_strength_matrix(self):
        """
        Build the synaptic strength matrix.
        """
        inhibitory_scaling_factor = torch.tensor([
            -num_excitatory / num_inhibitory
            for num_excitatory, num_inhibitory in zip(
                self.population_sizes[::2], self.population_sizes[1::2])
        ])

        synaptic_strength_column = torch.ones(self.num_populations) * self.baseline_synaptic_strength
        synaptic_strength_column[1::2] = inhibitory_scaling_factor * self.baseline_synaptic_strength

        self.recurrent_synaptic_strength = torch.tile(
            synaptic_strength_column, (self.num_populations, 1)) * self.internal_mask

    def _build_all_weights(self):
        """
        Build recurrent, background, external from synapse counts and synaptic strengths.
        """
        recurr_counts = self.recurrent_synapse_counts
        recurr_strength = self.recurrent_synaptic_strength
        self.recurrent_weights = self.recurrent_synapse_counts * self.recurrent_synaptic_strength
        recurr_weights = self.recurrent_weights
        background_weights = self.background_synapse_counts * self.baseline_synaptic_strength # device
        self.register_buffer("background_weights", background_weights) # device

    def _make_in_ex_masks(self, num_columns):
        """
        Make an internal mask with ones for within column connections
        and an external mask with ones for across column connections.
        """
        column_size = self.num_populations // num_columns  # will likely always be 8

        mask = torch.zeros(self.num_populations, self.num_populations)

        for i in range(0, self.num_populations, column_size):
            idx1 = i
            idx2 = i + column_size
            mask[idx1:idx2, idx1:idx2] = 1.0

        self.internal_mask = mask
        self.external_mask = 1 - mask



class ColumnNetwork(torch.nn.Module):

    '''
    Concatenates a number of areas (each consisting of a number
    of columns) to form a larger network. Within an area, only
    lateral connections between columns are allowed. Across areas
    only feedforward connections are allowed.
    '''

    def __init__(self, model_parameters, network_dict):
        super().__init__()

        self.noise_type = "diagonal"  # sde params
        self.sde_type = "ito"

        self._initialize_areas(model_parameters, network_dict)

        self.network_as_area = ColumnArea(model_parameters, 'v1', sum(network_dict['nr_columns_per_area']))
        self.nr_input_units = network_dict['nr_input_units']
        self.nr_columns_per_area = network_dict['nr_columns_per_area']
        self.nr_areas = network_dict['nr_areas']

        self._initialize_masks(model_parameters)
        self._initialize_feedforward_weights(model_parameters)
        self._initialize_feedback_weights(model_parameters)
        self._initialize_input_weights(model_parameters)
        self._initialize_output_weights(model_parameters)
        self._initialize_lateral_weights(model_parameters)

    def _initialize_areas(self, model_parameters, network_dict):
        '''
        Initialize the areas as ColumnArea objects.
        '''
        self.areas = nn.ModuleDict({})
        for area_idx in range(network_dict['nr_areas']):

            area_name = network_dict['areas'][area_idx]
            num_columns = network_dict['nr_columns_per_area'][area_idx]

            area = ColumnArea(model_parameters, area_name, num_columns)
            self.areas[str(area_idx)] = area

    def _initialize_masks(self, model_parameters):
        '''
        Binary masks to select only legal connections between populations,
        based on the nature of the connection.
        '''
        masks = model_parameters['connection_masks']

        self.input_mask = torch.tensor(masks['input'])
        self.output_mask = torch.tensor(masks['output'])
        self.feedforward_mask = torch.tensor(masks['feedforward'])
        self.feedback_mask = torch.tensor(masks['feedback'])
        self.lateral_mask = torch.tensor(masks['lateral'])

    def make_receptive_fields_v1_vanilla(self, fully_connected_input_mask, receptive_field_size=3, stride=1):
        '''
        Returns receptive field masks for V1 columns. No pre-specified
        filter shape, i.e. vanilla.
        '''
        num_input_pops, len_input_image = fully_connected_input_mask.shape
        num_input_cols = num_input_pops // 8  # eight populations = 1 column
        shape_input_image = int(np.sqrt(len_input_image))  # assumes the image shape is perfectly square (x_shape==y_shape)

        # Determine how many receptive fields we can have, given the image shape, receptive field size and stride
        nr_receptive_fields = ((shape_input_image - receptive_field_size + 1) // stride)**2
        # Then, determine how many columns will receive the identical receptive field
        nr_cols_per_receptive_field = num_input_cols // nr_receptive_fields

        # All receptive fields should have the exact same number of target columns
        assert num_input_cols % nr_receptive_fields == 0, \
            f"The number of columns in the first area ({num_input_cols}) can not be divided by the number of receptive fields ({nr_receptive_fields})."

        example_image = torch.tensor([  [ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
                                        [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
                                        [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
                                        [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
                                        [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
                                        [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
                                        [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
                                        [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])

        # Loop through the image indices to fill in the receptive field mask
        receptive_field_mask = torch.zeros(num_input_cols, len_input_image)

        col_idx = 0
        end = shape_input_image - receptive_field_size + 1

        for i in range(0, end, stride):
            for j in range(0, end, stride):
                for k in range(nr_cols_per_receptive_field):  # assign the same receptive field to n columns
                    image = torch.zeros(shape_input_image, shape_input_image)
                    image[i:i+receptive_field_size, j:j+receptive_field_size] = 1.0  # set all receptive field indices to 1
                    receptive_field_mask[col_idx, :] = image.flatten()
                    col_idx += 1

        # Extent receptive field mask from x_shape=nr_columns to x_shape=nr_populations
        receptive_field_mask_extended = receptive_field_mask.repeat_interleave(8, dim=0)

        return receptive_field_mask_extended * fully_connected_input_mask

    def make_receptive_fields_v1_hori_verti(self, fully_connected_input_mask, receptive_field_size=3, stride=1):
        '''
        Returns receptive field masks for V1 columns. The filter shapes
        are horizontal and vertical.
        '''
        num_input_pops, len_input_image = fully_connected_input_mask.shape
        num_input_cols = num_input_pops // 8  # eight populations = 1 column
        shape_input_image = int(np.sqrt(len_input_image))  # assumes the image shape is perfectly square (x_shape==y_shape)

        # Determine how many receptive fields we can have, given the image shape, receptive field size and stride
        nr_receptive_fields = ((shape_input_image - receptive_field_size + 1) // stride)**2
        # Then, determine how many columns will receive the identical receptive field
        nr_cols_per_receptive_field = num_input_cols // nr_receptive_fields

        # All receptive fields should have the exact same number of target columns
        assert num_input_cols % nr_receptive_fields == 0, \
            f"The number of columns in the first area ({num_input_cols}) can not be divided by the number of receptive fields ({nr_receptive_fields})."

        # Loop through the image indices to fill in the receptive field mask
        receptive_field_mask = torch.zeros(num_input_cols, len_input_image)

        col_idx = 0
        end = shape_input_image - receptive_field_size + 1

        for i in range(0, end, stride):
            for j in range(0, end, stride):

                # Hyper column: one horizontal and one vertical column
                image = torch.zeros(shape_input_image, shape_input_image)
                image[(i+1) : (i-1) + receptive_field_size, j:j + receptive_field_size] = 1.0  # horizontal
                receptive_field_mask[col_idx, :] = image.flatten()
                col_idx += 1

                image = torch.zeros(shape_input_image, shape_input_image)
                image[i:i + receptive_field_size, (j+1) : (j-1) + receptive_field_size] = 1.0  # vertical
                receptive_field_mask[col_idx, :] = image.flatten()
                col_idx += 1

        # Extent receptive field mask from x_shape=nr_columns to x_shape=nr_populations
        receptive_field_mask_extended = receptive_field_mask.repeat_interleave(8, dim=0)
        return receptive_field_mask_extended * fully_connected_input_mask

    def make_receptive_fields_v1_pca(self, fully_connected_input_mask, receptive_field_size=3, stride=1):
        '''
        Returns receptive field masks for V1 columns. The filter shapes
        are pre-defined and based on the learned weights of a standard nn.
        '''
        num_input_pops, len_input_image = fully_connected_input_mask.shape
        num_input_cols = num_input_pops // 8  # eight populations = 1 column
        shape_input_image = int(np.sqrt(len_input_image))  # assumes the image shape is perfectly square (x_shape==y_shape)

        # Determine how many receptive fields we can have, given the image shape, receptive field size and stride
        nr_receptive_fields = ((shape_input_image - receptive_field_size + 1) // stride)**2
        # Then, determine how many columns will receive the identical receptive field
        nr_cols_per_receptive_field = num_input_cols // nr_receptive_fields

        # All receptive fields should have the exact same number of target columns
        assert num_input_cols % nr_receptive_fields == 0, \
            f"The number of columns in the first area ({num_input_cols}) can not be divided by the number of receptive fields ({nr_receptive_fields})."

        # Loop through the image indices to fill in the receptive field mask
        receptive_field_mask = torch.zeros(num_input_cols, len_input_image)

        col_idx = 0
        end = shape_input_image - receptive_field_size + 1

        for i in range(0, end, stride):
            for j in range(0, end, stride):

                image = torch.zeros(shape_input_image, shape_input_image)
                image[i:i + receptive_field_size, j:j + receptive_field_size] = torch.tensor(
                    [[0., 0.1023325, 0.39980023],
                     [0., 0.04002324, 0.40573733],
                     [0., 0., 0.42890931]]) * 2.0
                receptive_field_mask[col_idx, :] = image.flatten()
                col_idx += 1

                image = torch.zeros(shape_input_image, shape_input_image)
                image[i:i + receptive_field_size, j:j + receptive_field_size] = torch.tensor(
                    [[0., 0., 0.22228338],
                     [0.,  0.2260077 ,  0.03974434],
                     [ 0.72426383, 0., 0.]]) * 2.0
                receptive_field_mask[col_idx, :] = image.flatten()
                col_idx += 1

        # Extent receptive field mask from x_shape=nr_columns to x_shape=nr_populations
        receptive_field_mask_extended = receptive_field_mask.repeat_interleave(8, dim=0)
        return receptive_field_mask_extended * fully_connected_input_mask

    def _initialize_input_weights(self, model_parameters):
        '''
        Initialize learnable input weights to weight the input going into the first area.
        '''
        first_area = self.areas['0']

        size_source = self.nr_input_units
        size_target = first_area.num_columns

        input_init = torch.tensor(model_parameters['connection_inits']['input'])
        input_init = torch.tile(input_init, (size_target, size_source))

        std_W = 1.0
        rand_input_weights = abs(torch.normal(mean=input_init, std=std_W))
        # rand_input_weights = input_init * first_area.baseline_synaptic_strength * 0.7  # for pca filters
        rand_input_weights = rand_input_weights * first_area.baseline_synaptic_strength * 0.2  # for learning from scratch

        input_mask = torch.tile(self.input_mask, (size_target, size_source))
        # Make 3x3 receptive fields
        # input_mask = self.make_receptive_fields_v1_hori_verti(input_mask)
        # input_mask = self.make_receptive_fields_v1_pca(input_mask)
        input_mask = self.make_receptive_fields_v1_vanilla(input_mask)
        first_area.register_buffer('input_mask', input_mask)

        rand_input_weights = rand_input_weights * input_mask
        first_area.input_weights = nn.Parameter(rand_input_weights, requires_grad=True)

    def _initialize_feedforward_weights(self, model_parameters):
        '''
        Initialize the feedforward weights between each set of areas as learnable weights.
        Attach the weights to the target area.
        '''
        for area_idx, area in self.areas.items():
            if area_idx != '0':  # first area gets no ff input

                size_source = self.nr_columns_per_area[int(area_idx) - 1]
                size_target = self.nr_columns_per_area[int(area_idx)]

                ff_init = torch.tensor(model_parameters['connection_inits']['feedforward'])
                ff_init = torch.tile(ff_init, (size_target, size_source))

                std_W = 1.0
                rand_ff_weights = abs(torch.normal(mean=ff_init, std=std_W))
                rand_ff_weights = rand_ff_weights * area.baseline_synaptic_strength * 0.7  # scale down in case of padding!

                ff_mask = torch.tile(self.feedforward_mask, (size_target, size_source))
                area.register_buffer('feedforward_mask', ff_mask)

                rand_ff_weights = rand_ff_weights * ff_mask
                area.feedforward_weights = nn.Parameter(rand_ff_weights, requires_grad=True)

    def _initialize_feedback_weights(self, model_parameters):
        '''
        Initialize the feedback weights between each set of areas as learnable weights.
        Attach the weights to the target area.
        '''
        for area_idx, area in self.areas.items():
            if int(area_idx) != (len(self.areas) - 1):  # last area gets no fb

                size_source = self.nr_columns_per_area[int(area_idx) + 1]
                size_target = self.nr_columns_per_area[int(area_idx)]

                fb_init = torch.tensor(model_parameters['connection_inits']['feedback'])
                fb_init = torch.tile(fb_init, (size_target, size_source))

                std_W = 1.0
                rand_fb_weights = abs(torch.normal(mean=fb_init, std=std_W))
                rand_fb_weights = fb_init * area.baseline_synaptic_strength * 0.0

                fb_mask = torch.tile(self.feedback_mask, (size_target, size_source))
                area.feedback_mask = fb_mask

                rand_fb_weights = rand_fb_weights * fb_mask
                area.feedback_weights = nn.Parameter(rand_fb_weights, requires_grad=False) # NOT TRAINING NOW

    def _initialize_output_weights(self, model_parameters):
        '''
        Initialize learnable output weights that can be used to read out
        the firing rates of the final column as a means of classification.
        '''
        key_last_area = str(len(self.areas)-1)
        size_source = self.areas[key_last_area].num_columns

        output_init = torch.tensor(model_parameters['connection_inits']['output'])
        output_init = torch.tile(output_init, (size_source,))

        # std_W = 0.001
        # rand_output_weights = abs(torch.normal(mean=output_init, std=std_W))
        # rand_output_weights *= torch.tile(self.output_mask, (size_source,))

        self.output_weights = nn.Parameter(output_init, requires_grad=False)  # NOT TRAINING NOW

    def _initialize_lateral_weights(self, model_parameters):
        '''
        Initialize learnable lateral inhibition weights between each
        set of V1 columns (i.e. together forming a hyper column) and
        between all the output columns.
        '''
        for area_idx, area in self.areas.items():
            inner_weights = area.recurrent_weights * area.internal_mask  # set any existing external connectivity to zero
            area.register_buffer('inner_weights', inner_weights)

            # Initialize lateral weights (860 for V1; 915 for V2)
            lateral_init = torch.ones((area.num_populations, area.num_populations)) * 0.1
            lateral_init = lateral_init

            # Reshape lateral mask
            lateral_mask = torch.tile(self.lateral_mask, (area.num_columns, area.num_columns)) * area.external_mask

            # First area gets lateral weights between each pair of filters with the same receptive field
            if area_idx == '0':
                # Make lateral mask follow hyper-column organization
                n = area.num_populations
                block = 8 * 2  # two columns in one hyper-column
                lateral_hyper_column_mask = torch.eye(n // block).repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
                lateral_mask = lateral_mask * lateral_hyper_column_mask

            # Store mask and weights in area
            area.lateral_mask = lateral_mask
            lateral_weights = lateral_init * lateral_mask

            area.lateral_weights = nn.Parameter(lateral_weights, requires_grad=False)  # NOT TRAINING NOW

    def set_time_vec(self, time_vec):
        '''
        Set the time_vec as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.time_vec = time_vec

    def set_stim(self, stim):
        '''
        Set the stimulus as a mutable attribute. This is necessary because
        torchsde does not allow any extra parameters other than t, y0.
        '''
        self.stim = stim

    def get_device(self):
        '''
        Gets the network's current device, based on the first area's
        feedforward weights.
        '''
        return self.areas['0'].input_weights.device

    def partition_firing_rates(self, firing_rate):
        '''
        Organizes the firing rates into a dict of separate areas.
        This allows easy access to previous area's firing rates.
        '''
        fr_per_area = {}
        idx = 0
        for area_idx, area in self.areas.items():
            fr_area = firing_rate[:, idx : idx + area.num_populations]
            fr_per_area[area_idx] = fr_area
            idx = idx + area.num_populations
        return fr_per_area

    def compute_currents(self, ext_ff_rate, fr_per_area, t):
        '''
        Compute the current for each area separately. The total current
        consists of feedforward current (stimulus-driven and/or from other
        brain areas), background current and recurrent current.
        '''
        total_current = torch.Tensor().to(self.get_device())

        for area_idx, area in self.areas.items():

            # Compute feedforward current of each area, based on
            # area=0: external input or area>0: the previous area's firing rate
            feedforward_current = torch.zeros(area.num_populations)
            if area_idx == '0':
                feedforward_current = torch.matmul(ext_ff_rate, area.input_weights.T)
            elif area_idx > '0':  # subsequent areas receive previous area's firing rate
                idx_prev_area = str(int(area_idx) - 1)
                prev_area_fr = fr_per_area[idx_prev_area]
                feedforward_current = torch.matmul(prev_area_fr, area.feedforward_weights.T)

            # # Compute feedback current
            # feedback_current = torch.zeros(area.num_populations)
            # if int(area_idx) < (len(self.areas) - 1):  # only last area has no fb weights, so skip that one
            #     key_next_area = str(int(area_idx) + 1)
            #     next_area_fr = fr_per_area[key_next_area]
            #     feedback_current = torch.matmul(area.feedback_weights, next_area_fr)

            # Compute recurrent current
            recurrent_current = torch.matmul(fr_per_area[area_idx], area.inner_weights.T)
            lateral_current = torch.matmul(fr_per_area[area_idx], area.lateral_weights.T)

            # Background current
            background_current = torch.tile(area.background_drive, (ext_ff_rate.shape[0], 1)) * area.background_weights

            # Total current of this area
            total_current_area = (feedforward_current +
                                  lateral_current +
                                  recurrent_current +
                                  background_current) * area.synapse_time_constant
            total_current = torch.cat((total_current, total_current_area), dim=1)

        return total_current

    def forward(self, t, state):
        '''
        State dynamics updating the membrane potential and adaptation;
        ODE should learn these dynamics and update the weights accordingly.
        '''

        # Unpack the state (membrane, adaptation)
        mem_adap_split = state.shape[1] // 2
        membrane_potential, adaptation = state[:, :mem_adap_split], state[:, mem_adap_split:]

        # Partition firing rate per area
        firing_rate = compute_firing_rate(membrane_potential - adaptation)
        fr_per_area = self.partition_firing_rates(firing_rate)

        # If over half the time has passed, present the stimulus
        ext_ff_rate = torch.zeros_like(self.stim)
        if t > self.time_vec[len(self.time_vec) // 2]:
            ext_ff_rate = self.stim

        # Compute input current
        total_current = self.compute_currents(ext_ff_rate, fr_per_area, t)

        # Compute derivative membrane potential and adaptation
        delta_membrane_potential = (-membrane_potential +
            total_current * self.network_as_area.resistance) / self.network_as_area.membrane_time_constant
        delta_adaptation = (-adaptation + self.network_as_area.adaptation_strength *
                            firing_rate) / self.network_as_area.adapt_time_constant

        state = torch.concat((delta_membrane_potential, delta_adaptation), dim=1)
        return state

    def diffusion(self, t, y):
        '''
        Diffusion function used by SDE, noise is only applied
        to membrane potential.
        '''
        noise_std = 3.0
        g = torch.zeros_like(y)
        split = (len(y[0]) // 2)
        g[:, :split] = noise_std
        return g
