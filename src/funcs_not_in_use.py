

'''
Possible V2 receptive fields
'''
def make_receptive_fields_v2(self, fully_connected_mask, receptive_field_size=3, stride=1, nr_source_cols_same_rf=1):
    num_target_pops, num_source_pops = fully_connected_mask.shape
    num_target_cols = num_target_pops // 8  # eight populations = 1 column
    num_source_cols = num_source_pops // 8
    size_feature_map = int(np.sqrt(num_source_cols // nr_source_cols_same_rf))

    # Determine how many receptive fields we can have, given the feature map shape, receptive field size and stride
    nr_receptive_fields = ((size_feature_map - receptive_field_size + 1) // stride) ** 2
    # Then, determine how many columns will receive the identical receptive field
    nr_cols_per_receptive_field = num_target_cols // nr_receptive_fields

    # All receptive fields should have the exact same number of target columns
    assert num_target_cols % nr_receptive_fields == 0, \
        f"The number of columns in the target area ({num_target_cols}) can not be divided by the number of receptive fields ({nr_receptive_fields})."

    # Loop through the feature map indices to fill in the receptive field mask
    receptive_field_mask = torch.zeros(num_target_cols, num_source_cols)

    col_idx = 0
    end = size_feature_map - receptive_field_size + 1

    for i in range(0, end, stride):
        for j in range(0, end, stride):
            for k in range(nr_cols_per_receptive_field):  # assign the same receptive field to n columns
                image = torch.zeros(size_feature_map, size_feature_map)
                image[i:i + receptive_field_size,
                j:j + receptive_field_size] = 1.0  # set all receptive field indices to 1
                flattened_image = image.flatten()
                flattened_image_extended = flattened_image.repeat_interleave((nr_source_cols_same_rf), dim=0)
                receptive_field_mask[col_idx, :] = flattened_image_extended
                col_idx += 1

    # Extent receptive field mask from x_shape=nr_columns to x_shape=nr_populations - and the same for dimension y
    receptive_field_mask_extended = receptive_field_mask.repeat_interleave((8), dim=0)
    receptive_field_mask_extended = receptive_field_mask_extended.repeat_interleave((8), dim=1)

    return receptive_field_mask_extended * fully_connected_mask



'''
Code for input filters targeting inhibitory populations
'''
def make_receptive_fields_v1_hori_verti(self, fully_connected_input_mask, receptive_field_size=3, stride=1):

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
            image[(i+1):(i-1) + receptive_field_size, j:j + receptive_field_size] = 1.0  # horizontal
            image[(i+0):(i-2) + receptive_field_size, j:j + receptive_field_size] = -1.0
            image[(i+2):(i-0) + receptive_field_size, j:j + receptive_field_size] = -1.0
            receptive_field_mask[col_idx, :] = image.flatten()
            col_idx += 1

            image = torch.zeros(shape_input_image, shape_input_image)
            image[i:i + receptive_field_size, (j+1):(j-1) + receptive_field_size] = 1.0  # vertical
            image[i:i + receptive_field_size, (j+0):(j-2) + receptive_field_size] = -1.0
            image[i:i + receptive_field_size, (j+2):(j-0) + receptive_field_size] = -1.0
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

    input_init = (torch.tensor(model_parameters['connection_inits']['input']))
    input_init = input_init * first_area.baseline_synaptic_strength * 0.7
    input_init_excitatory = torch.tile(input_init, (size_target, size_source))

    # std_W = 1.0
    # rand_input_weights = abs(torch.normal(mean=input_init, std=std_W))

    # Make 3x3 receptive fields
    input_mask = torch.tile(self.input_mask, (size_target, size_source))
    input_mask = self.make_receptive_fields_v1_hori_verti(input_mask)
    first_area.input_mask = abs(input_mask)

    # Apply receptive field input mask to weights
    input_init_excitatory = input_init_excitatory * input_mask
    input_init_excitatory[input_init_excitatory < 0] = 0  # set everything below zero to zero
    # Also apply receptive field inhibition-targeted connections
    exci_inhi_swapped = torch.zeros(input_init.shape)
    exci_inhi_swapped[2], exci_inhi_swapped[3] = input_init[3], input_init[2]
    exci_inhi_swapped_all_cols = torch.tile(exci_inhi_swapped, (size_target, size_source))
    input_init_inhibition = exci_inhi_swapped_all_cols * (input_mask * -1.0)
    input_init_inhibition[input_init_inhibition < 0] = 0  # set everything below zero to zero

    # Add excitatory- and inhibitory-targeting weights together to form receptive fields
    # input_weights = input_init_excitatory + input_init_inhibition
    input_weights = input_init_excitatory
    first_area.input_weights = nn.Parameter(input_weights, requires_grad=False) # NOT TRAINING NOW
