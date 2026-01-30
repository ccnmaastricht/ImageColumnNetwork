import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pprint import pprint
import torch



def normalize_filters(filters):
    # zero mean
    filters = filters - filters.mean(axis=1, keepdims=True)
    # unit norm
    norms = np.linalg.norm(filters, axis=1, keepdims=True) + 1e-8
    return filters / norms

def pca_consensus_filters(filters, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(filters)
    components = pca.components_  # shape (2, 9)
    return components

def find_most_informative_filters():
    filters = np.array([[[-0.3140, 0.2041, 0.4538],
                         [-0.2631, 0.2632, 0.0108],
                         [0.2924, -0.0992, 0.1592]],
                        [[0.3669, -0.1071, 0.1500],
                         [0.2878, 0.4603, 0.2913],
                         [-0.1320, -0.0405, -0.0271]],
                        [[-0.5031, -0.0029, 0.4587],
                         [-0.0180, 0.3319, 0.3157],
                         [-0.1121, 0.0629, 0.3657]],
                        [[0.2142, 0.1402, -0.2877],
                         [0.3060, 0.0339, -0.1231],
                         [0.1616, 0.3759, 0.2522]],
                        [[0.2570, 0.0336, 0.0464],
                         [0.1991, 0.1149, -0.0079],
                         [0.4579, 0.0603, -0.2710]],
                        [[-0.1878, 0.0898, 0.1607],
                         [-0.0346, 0.3869, 0.2561],
                         [0.2673, -0.1651, 0.3940]],
                        [[0.1783, 0.3383, -0.2344],
                         [0.3161, 0.2030, -0.1213],
                         [0.0062, 0.4274, -0.1905]],
                        [[0.4410, -0.0963, 0.2415],
                         [0.4228, -0.0237, -0.3558],
                         [0.1510, 0.4727, -0.2090]],
                        [[-0.2240, -0.3655, -0.1342],
                         [-0.3770, 0.1832, 0.3482],
                         [0.2484, 0.2924, -0.1345]],
                        [[0.3482, 0.1449, 0.0474],
                         [-0.0575, 0.4215, 0.1874],
                         [0.4192, 0.1110, -0.2045]],
                        [[0.3023, 0.1919, 0.2623],
                         [-0.0507, -0.1391, 0.4502],
                         [-0.3274, 0.2049, 0.2855]],
                        [[0.2163, -0.4526, -0.1959],
                         [-0.0133, -0.1732, 0.0078],
                         [0.2712, 0.2197, 0.0463]],
                        [[0.3874, 0.3256, -0.2704],
                         [0.2765, 0.2165, -0.2129],
                         [0.1703, 0.3640, -0.1592]],
                        [[0.4687, 0.3234, 0.1848],
                         [0.4398, 0.2943, 0.2519],
                         [-0.0694, 0.1582, 0.3930]],
                        [[0.0776, -0.0735, -0.3411],
                         [0.4109, 0.0831, -0.0348],
                         [0.2324, 0.4686, -0.0354]],
                        [[0.3559, 0.1481, 0.3422],
                         [0.1186, -0.4100, 0.3007],
                         [0.2603, -0.4047, 0.3580]],
                        [[0.1676, 0.3080, -0.0717],
                         [-0.1929, 0.1643, 0.3577],
                         [-0.4281, -0.0362, 0.3811]],
                        [[0.0212, 0.0857, 0.3198],
                         [-0.2775, 0.0258, 0.1626],
                         [-0.2441, -0.0383, 0.4157]],
                        [[0.1522, -0.1249, -0.0932],
                         [0.0863, -0.0268, -0.4238],
                         [0.4077, 0.1157, -0.0288]],
                        [[0.1695, -0.1806, 0.1579],
                         [0.2403, 0.2501, 0.2671],
                         [0.1603, 0.4318, 0.1649]]
                        ])

    filters = filters.reshape(filters.shape[0], -1)  # -> (20, 9)

    filters = normalize_filters(filters)

    components = pca_consensus_filters(filters, n_components=2)

    # reshape to 3×3 kernels
    filter1 = components[0].reshape(3, 3)
    filter2 = components[1].reshape(3, 3)

    def enforce_positive_center(kernel):
        if kernel[1, 1] < 0:
            return -kernel
        return kernel

    filter1 = enforce_positive_center(filter1)
    filter2 = enforce_positive_center(filter2)

    pprint(filter1)
    pprint(filter2)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(filter1, cmap='gray')
    axs[0].set_title('PCA Filter 1')
    axs[1].imshow(filter2, cmap='gray')
    axs[1].set_title('PCA Filter 2')
    plt.show()

    '''
    These are the most informative filters according to PCA - for classes 0 and 1
    array([[-0.28060338,  0.1023325 ,  0.39980023],
           [-0.36143616,  0.04002324,  0.40573733],
           [-0.35770524, -0.37705782,  0.42890931]])
    array([[-0.41795629, -0.27462643,  0.22228338],
           [-0.30471112,  0.2260077 ,  0.03974434],
           [ 0.72426383, -0.04707407, -0.16793134]])
    '''


