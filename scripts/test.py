import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import torch



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
    pca_padding_lat_in = np.array([0.55754, 0.43933, 0.32062, 0.22603, 0.16665, 0.12603, 0.09817, ])

    plt.plot(first_results, label='baseline model (h/v)')
    plt.plot(lateral_inhibition, label='learnable lateral inhibition')
    plt.plot(fixed_lat_in, label='fixed lateral inhibition')
    plt.plot(standard_nn_filters, label='using standard nn filters')
    plt.legend()
    plt.show()

    plt.plot(two_digits_loss, label='two digits loss')
    plt.plot(two_digits_acc, label='two digits accuracy', linestyle='--')
    plt.plot(four_digits_loss, label='four digits loss')
    plt.plot(four_digits_acc, label='four digits accuracy', linestyle='--')
    plt.legend()
    plt.show()

    plt.plot(standard_nn_filters, label='standard nn filters')
    plt.plot(pca_padding, label='+ padding')
    plt.plot(pca_padding_lat_in, label='+ lateral inhibition')
    plt.legend()
    plt.show()



def experiment_with_loss_functions():
    labels = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1])
    model_predictions_1 = torch.tensor([[1.9, 1.8],
                                      [1.5, 18.5],
                                      [1.4, 21.0],
                                      [3.1, 2.9],
                                      [18.0, 3.6],
                                      [2.6, 24.0],
                                      [23.5, 18.0],
                                      [1.0, 21.5]])
    model_predictions_2 = torch.tensor([[21.9, 1.8],
                                      [1.5, 18.5],
                                      [1.4, 21.0],
                                      [23.1, 2.9],
                                      [18.0, 3.6],
                                      [2.6, 24.0],
                                      [23.5, 1.0],
                                      [1.0, 21.5]])
    model_predictions_3 = torch.tensor([[1.9, 1.8],
                                      [1.5, 1.9],
                                      [1.4, 3.0],
                                      [3.1, 2.9],
                                      [2.0, 1.6],
                                      [2.6, 2.8],
                                      [2.5, 1.0],
                                      [1.0, 1.5]])
    targets =             torch.tensor([[20., 0.],
                                       [0., 20.],
                                       [0., 20.],
                                       [20., 0.],
                                       [20., 0.],
                                       [0., 20.],
                                       [20., 0.],
                                       [0., 20.],
                                       ])

    # suppress non-target firing
    mask = torch.nn.functional.one_hot(labels, num_classes=2)
    bloop = (1 - mask)
    bloopie = (1 - mask) * model_predictions_1
    suppression = ((1 - mask) * model_predictions_1).mean()

    bleppie = torch.nn.functional.one_hot(labels, num_classes=2) * 20.
    print(bleppie)

    criterion = torch.nn.CrossEntropyLoss()
    print(criterion(model_predictions_1, labels))
    print(criterion(model_predictions_2, labels))
    print(criterion(model_predictions_3, labels))

    print(torch.mean(abs(model_predictions_1 - targets)))
    print(torch.mean(abs(model_predictions_2 - targets)))
    print(torch.mean(abs(model_predictions_3 - targets)))


plot_losses()

