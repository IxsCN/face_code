# from scipy.stats import wasserstein_distance
#
# y = wasserstein_distance([0, 1, 3], [0, 3, 1])
# print(y)
#
# y2 = wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4, 0, 0],
#                      [1.4, 0.9, 3.1, 7.2], [3.2, 3.5, 0, 0])
#
# print(y2)

# import numpy as np
# def l2_norm_array_by_matrix_norm(data):
#     tmp = data * data
#     return data / np.sqrt(sum(sum(tmp)))
#
#
# ImbalancedDatasetSampling = np.array([[j+i for j in range(2)] for i in range(2)])
# print(ImbalancedDatasetSampling)
# print(l2_norm_array_by_matrix_norm(ImbalancedDatasetSampling))

from scipy import optimize
from autograd import grad
import autograd
autograd.n

rst = optimize.minimize(proj_error_function,
                                f3d,
                                args=(proj_matrics, row['p2d'], row['score']),
                                method='BFGS', fprime=grad_tanh)