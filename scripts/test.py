import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch



def heatmap_model_output():

    padding_epoch_11 = np.array([[ 2.1081,  1.7095,  1.0000],
        [18.2088, 23.7431,  1.0000],
        [21.0534,  2.7949,  0.0000],
        [ 7.0531,  1.6512,  0.0000],
        [ 1.5573, 29.1697,  1.0000],
        [ 0.8974, 27.4820,  1.0000],
        [ 1.6300, 28.3759,  1.0000],
        [ 0.8135,  3.8737,  1.0000],
        [20.2409,  2.2814,  0.0000],
        [20.3450,  2.0996,  0.0000],
        [10.3562,  1.6422,  0.0000],
        [20.1825,  2.6212,  0.0000],
        [ 1.8199, 27.0299,  1.0000],
        [ 2.6044,  1.2652,  0.0000],
        [ 3.8498,  1.4647,  0.0000],
        [ 1.8774, 19.9137,  1.0000],
        [27.4397, 16.2833,  0.0000],
        [ 0.6259, 11.2130,  1.0000],
        [ 1.2808,  1.8132,  1.0000],
        [22.1968, 12.2289,  0.0000],
        [18.4473,  1.8687,  0.0000],
        [ 0.9525, 21.1157,  1.0000],
        [18.8739,  2.0199,  0.0000],
        [25.8977, 16.6268,  0.0000],
        [ 1.8636, 25.1561,  1.0000],
        [23.8223, 13.9558,  0.0000],
        [ 2.2672,  1.2489,  0.0000],
        [ 0.6157, 26.5556,  1.0000],
        [23.4143,  3.8867,  0.0000],
        [13.1388, 38.8211,  1.0000],
        [ 0.7004, 27.3994,  1.0000],
        [ 3.1039,  1.3288,  0.0000],
        [ 2.5266,  1.2578,  0.0000],
        [ 1.6956, 16.5246,  1.0000],
        [ 2.1970, 25.1176,  1.0000],
        [19.4149,  2.2874,  0.0000]])
    suppression_epoch_11 = np.array([[ 1.8743,  1.1645,  1.0000],
        [ 2.4095, 19.3593,  1.0000],
        [22.3111,  1.3426,  0.0000],
        [17.3780,  0.9792,  0.0000],
        [ 0.5770, 29.3470,  1.0000],
        [ 0.3104, 29.7568,  1.0000],
        [ 0.5826, 30.0441,  1.0000],
        [ 0.4564,  7.7095,  1.0000],
        [21.7768,  1.1886,  0.0000],
        [22.1619,  1.0816,  0.0000],
        [18.5994,  0.9871,  0.0000],
        [20.8457,  1.2753,  0.0000],
        [ 0.7044, 27.8216,  1.0000],
        [ 3.3505,  0.8100,  0.0000],
        [15.4991,  0.9045,  0.0000],
        [ 1.0787, 18.2675,  1.0000],
        [28.8347,  1.2424,  0.0000],
        [ 0.2992, 20.6867,  1.0000],
        [ 0.9898,  1.5690,  1.0000],
        [22.2260,  1.7708,  0.0000],
        [20.0540,  1.0270,  0.0000],
        [ 0.4344, 22.8888,  1.0000],
        [20.1719,  1.1094,  0.0000],
        [27.3045,  1.6909,  0.0000],
        [ 0.8360, 26.5077,  1.0000],
        [23.8946,  1.5122,  0.0000],
        [ 2.6358,  0.8169,  0.0000],
        [ 0.2075, 30.0421,  1.0000],
        [25.0462,  1.5057,  0.0000],
        [ 0.6174, 40.8197,  1.0000],
        [ 0.2439, 30.9161,  1.0000],
        [ 5.5054,  0.8221,  0.0000],
        [ 3.2117,  0.8182,  0.0000],
        [ 1.0067, 14.1556,  1.0000],
        [ 1.0240, 24.7990,  1.0000],
        [20.6746,  1.2504,  0.0000]])
    padding_latin_epoch_06 = np.array([[ 1.5635,  1.2854,  1.0000],
        [ 3.5328, 19.7534,  1.0000],
        [16.9616,  1.8492,  0.0000],
        [ 2.5491,  1.2514,  0.0000],
        [ 1.4939, 24.5783,  1.0000],
        [ 0.9110, 23.3081,  1.0000],
        [ 1.5090, 24.7187,  1.0000],
        [ 0.7663,  2.0110,  1.0000],
        [13.0834,  1.5997,  0.0000],
        [13.4064,  1.4904,  0.0000],
        [ 2.6300,  1.2501,  0.0000],
        [11.9784,  1.7188,  0.0000],
        [ 1.5907, 23.2561,  1.0000],
        [ 1.8144,  1.0024,  0.0000],
        [ 2.2909,  1.1376,  0.0000],
        [ 1.6289,  4.2085,  1.0000],
        [24.2685,  2.3618,  0.0000],
        [ 0.6230,  2.3770,  1.0000],
        [ 1.1048,  1.3164,  1.0000],
        [19.1522,  2.4935,  0.0000],
        [ 3.5697,  1.3579,  0.0000],
        [ 0.9108,  9.4826,  1.0000],
        [ 3.8370,  1.4617,  0.0000],
        [22.5890,  2.7544,  0.0000],
        [ 1.7338, 21.1056,  1.0000],
        [20.5447,  2.4410,  0.0000],
        [ 1.6413,  0.9905,  0.0000],
        [ 0.6600, 21.2538,  1.0000],
        [20.0637,  2.1940,  0.0000],
        [13.7534, 33.6856,  1.0000],
        [ 0.7527, 23.1514,  1.0000],
        [ 2.0506,  1.0398,  0.0000],
        [ 1.7823,  0.9984,  0.0000],
        [ 1.4701,  3.1466,  1.0000],
        [ 1.9850, 21.0019,  1.0000],
        [ 4.8653,  1.6202,  0.0000]])
    suppression_epoch_05 = np.array([[ 1.2832,  0.9400,  1.0000],
        [ 1.4346,  4.4235,  1.0000],
        [12.5137,  1.1239,  0.0000],
        [ 2.3275,  0.8443,  0.0000],
        [ 0.7425, 24.4933,  1.0000],
        [ 0.4633, 25.1429,  1.0000],
        [ 0.6924, 25.6142,  1.0000],
        [ 0.5168,  2.0538,  1.0000],
        [ 5.4090,  0.9980,  0.0000],
        [ 7.4113,  0.9203,  0.0000],
        [ 2.4349,  0.8406,  0.0000],
        [ 4.0926,  1.0499,  0.0000],
        [ 0.7522, 23.3927,  1.0000],
        [ 1.7322,  0.7086,  0.0000],
        [ 2.1774,  0.7827,  0.0000],
        [ 1.0835,  3.0222,  1.0000],
        [22.9414,  1.0260,  0.0000],
        [ 0.3876,  3.7225,  1.0000],
        [ 0.8819,  1.1088,  1.0000],
        [17.4698,  1.3638,  0.0000],
        [ 3.0826,  0.8686,  0.0000],
        [ 0.5521, 15.8217,  1.0000],
        [ 3.2183,  0.9418,  0.0000],
        [21.6012,  1.3133,  0.0000],
        [ 0.9650, 20.8284,  1.0000],
        [18.3965,  1.1915,  0.0000],
        [ 1.5210,  0.7086,  0.0000],
        [ 0.3297, 24.6717,  1.0000],
        [19.3940,  1.2402,  0.0000],
        [ 0.9662, 34.8164,  1.0000],
        [ 0.3723, 25.4727,  1.0000],
        [ 1.9570,  0.7193,  0.0000],
        [ 1.6919,  0.7097,  0.0000],
        [ 1.0092,  2.5475,  1.0000],
        [ 1.1188, 19.8795,  1.0000],
        [ 3.7652,  1.0458,  0.0000]])


    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    heatmap1 = axes[0].imshow(suppression_epoch_11[:, 2:], cmap="magma", interpolation="nearest")
    fig.colorbar(heatmap1, ax=axes[0])
    axes[0].set_title('Labels')

    heatmap2 = axes[1].imshow(padding_epoch_11[:, :2], cmap="magma", interpolation="nearest")
    fig.colorbar(heatmap2, ax=axes[1])
    axes[1].set_title('FR after training with CE only')

    heatmap3 = axes[2].imshow(suppression_epoch_11[:, :2], cmap="magma", interpolation="nearest")
    fig.colorbar(heatmap3, ax=axes[2])
    axes[2].set_title('FR after training with suppression')

    plt.show()




def plot_losses():
    first_results = np.array([0.36819, 0.29233, 0.23789, 0.19832, 0.17293, 0.15922, 0.17718, 0.13544, 0.10962, 0.09414,
                              0.08137, 0.07381, 0.06910, 0.07517, 0.07555, 0.07035, 0.07172, 0.04900, 0.05166, 0.10520, 0.06123])
    lateral_inhibition = np.array([0.29813, 0.24475, 0.19827, 0.15619, 0.14935, 0.17384])
    fixed_lat_in = np.array([0.61933, 0.59293, 0.56937, 0.54642, 0.52594, 0.50350, 0.48058, 0.45684])
    standard_nn_filters = np.array([0.48396, 0.38025, 0.30133, 0.23103, 0.17630, 0.14610, 0.12705, 0.11318, 0.10259, 0.09445,
                                    0.08690, 0.07900, 0.07248, 0.06585, 0.06182, 0.06448, 0.06534, 0.06472, 0.06421, 0.06350, 0.06279])

    four_digits_loss = np.array([0.95601, 0.83898, 0.73341, 0.65373, 0.57863, 0.61715, 0.43527, 0.32707, 0.44571, 0.33031, 0.32502, 0.26905, 0.44826, 0.46390])
    four_digits_acc = np.array([0.83, 0.90, 0.86, 0.86, 0.85, 0.85, 0.88, 0.92, 0.89, 0.90, 0.88, 0.90, 0.88, 0.88])
    two_digits_loss = np.array([0.39998, 0.35447, 0.3240, 0.30238, 0.28452])
    two_digits_acc = np.array([0.97, 0.94, 0.97, 1.00, 1.00])

    pca_padding = np.array([0.55758, 0.43940, 0.32070, 0.22607, 0.16668, 0.12543, 0.09544, 0.07879, 0.07166, 0.06805,
                            0.06698, 0.06840])
    pca_padding_lat_in = np.array([0.55754, 0.43933, 0.32062, 0.22603, 0.16665, 0.12603, 0.09817])

    # This is on pca_padding_lat_in
    pca_mae = np.array([9.63104, 9.40829, 7.71851, 6.60577, 6.21687, 6.45653, 6.42071, 6.44609, 6.43730, 6.84118,
                        6.88520, 6.87842, 6.87177, 6.86233])
    pca_mae_ce = np.array([0.58435, 0.47558, 0.33174, 0.20121, 0.15953, 0.17874, 0.18029, 0.18184, 0.17827, 0.23398,
                           0.24048, 0.24028, 0.23970, 0.23838])
    pca_mae_acc = np.array([0.47, 0.81, 1.00, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 1.00, 1.00, 1.00, 1.00, 1.00])
    pca_supp = np.array([0.45075, 0.41481, 0.39346, 0.37423, 0.35702, 0.34010])
    pca_supp_ce = np.array([0.12848, 0.14334, 0.13948, 0.13223, 0.12619, 0.11824])
    pca_supp_acc = np.array([0.97, 0.97, 0.97, 0.97, 0.97, 0.97])

    # This is on pca_padding (no latin)
    pca_pad_supp_ce = np.array([0.07358, 0.07765, 0.07441, 0.07076, 0.06777, 0.06482, 0.06259, 0.05888, 0.05673, 0.05473, 0.05329, 0.05217])
    pca_pad_supp = np.array([0.57476, 0.42696, 0.39675, 0.37703, 0.36005, 0.34660, 0.33612, 0.33011, 0.32192, 0.31591, 0.30873, 0.30069])

    # plt.plot(first_results, label='baseline model (h/v)')
    # plt.plot(lateral_inhibition, label='learnable lateral inhibition')
    # plt.plot(fixed_lat_in, label='fixed lateral inhibition')
    # plt.plot(standard_nn_filters, label='using standard nn filters')
    # plt.legend()
    # plt.show()  # NN-learned filters perform well, similarly to horizontal/vertical baseline

    # plt.plot(standard_nn_filters, label='two digits loss')
    # plt.plot(two_digits_acc, label='two digits accuracy', linestyle='--')
    # plt.plot(four_digits_loss, label='four digits loss')
    # plt.plot(four_digits_acc, label='four digits accuracy', linestyle='--')
    # plt.legend()
    # plt.show()  # Four digits might be too difficult with this V1-V2 architecture

    # plt.plot(standard_nn_filters, label='standard nn filters')
    # plt.plot(pca_padding, label='+ padding')
    # plt.plot(pca_padding_lat_in, label='+ lateral inhibition')
    # plt.legend()
    # plt.show()  # With padding training is faster; with lateral inhibition is similarly fast

    # plt.plot(pca_mae, label='MAE')
    # plt.plot(pca_mae_ce, label='CE (training on MAE)')
    # plt.plot(pca_mae_acc, label='MAE accuracy', linestyle='--')
    # plt.legend()
    # plt.show()  # Verdict: terrible!

    plt.plot(np.concat([pca_padding_lat_in, pca_supp]), label='CE with suppression')
    plt.plot(np.concat([pca_padding_lat_in, pca_supp_ce]), label='CE')
    plt.legend()
    plt.show()  # Good, but fires rates were not really affected

    plt.plot(np.concat([pca_padding, pca_pad_supp]), label='CE with suppression')
    plt.plot(np.concat([pca_padding, pca_pad_supp_ce]), label='CE')
    plt.legend()
    plt.show()  # Great! Adding suppression to loss leads to lower CE and lower firing rates for the loser column!



def experiment_with_loss_functions():
    labels = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1])
    model_predictions = torch.tensor([[[1.9, 1.8],
                                      [1.5, 18.5],
                                      [1.4, 21.0],
                                      [3.1, 2.9],
                                      [18.0, 3.6],
                                      [2.6, 24.0],
                                      [23.5, 18.0],
                                      [1.0, 21.5]],
                                      [[31.9, 1.8],
                                       [1.5, 38.5],
                                       [1.4, 31.0],
                                       [33.1, 2.9],
                                       [38.0, 3.6],
                                       [2.6, 34.0],
                                       [33.5, 1.0],
                                       [1.0, 31.5]],
                                      [[1.9, 1.8],
                                       [1.5, 1.9],
                                       [1.4, 3.0],
                                       [3.1, 2.9],
                                       [2.0, 1.6],
                                       [2.6, 2.8],
                                       [2.5, 1.0],
                                       [1.0, 1.5]],
                                      [[31.9, 10.8],
                                       [21.5, 31.9],
                                       [10.4, 23.0],
                                       [34.1, 21.9],
                                       [25.0, 16.6],
                                       [20.6, 32.8],
                                       [29.5, 13.0],
                                       [17.0, 41.5]]
                                      ])
    targets =             torch.tensor([[20., 0.],
                                       [0., 20.],
                                       [0., 20.],
                                       [20., 0.],
                                       [20., 0.],
                                       [0., 20.],
                                       [20., 0.],
                                       [0., 20.],
                                       ])


    criterion = torch.nn.CrossEntropyLoss()

    for i in range(len(model_predictions)):
        print(i)
        preds = model_predictions[i]

        # suppress non-target firing
        mask = torch.nn.functional.one_hot(labels, num_classes=2)
        suppression = ((1 - mask) * preds).mean()

        # force 20 Hz difference between firing rates
        force_diff = torch.relu(20 - torch.mean(abs(preds[:, 0] - preds[:, 1])))

        print(criterion(preds, labels))
        print((0.5 * suppression))
        print((0.1 * suppression))
        # print((0.5 * force_diff))

        # print(torch.mean(abs(preds - targets)))


# plot_losses()
heatmap_model_output()

