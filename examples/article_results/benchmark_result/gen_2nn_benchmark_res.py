import numpy as np
from scipy.io import loadmat
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

from cmfsapy.dimension.fsa import fsa
from cmfsapy.dimension.correction import correct_estimates
from cmfsapy.dimension.fsa import get_dists_inds_ck
from cmfsapy.evaluation.mpe import compute_mpe, compute_pe, compute_p_error



def compute_2nn(X, boxsize=None, cut_perc=5):
    """2nn nearest neighbor dimension estimate

    :param X: data (n, dim)
    :param boxsize: circular boundary size
    :param cut_perc: cutoff percentage in the least square fit
    :return: dimension estimate
    """
    # get normalized distance of the nearest neighbor
    R, inds = get_dists_inds_ck(X, k=2, boxsize=boxsize)
    mu = np.sort(R[:, 2] / R[:, 1])

    # fit line to the distribution of normalized nn distances
    n = X.shape[0]
    cutoff = cut_perc / 100

    F = np.arange(1, n + 1) / n

    mu_ok = mu[F < (1-cutoff)]
    F_ok = F[F < (1-cutoff)]

    # least square fit
    x = np.log(mu_ok)
    y = - np.log(1 - F_ok)
    a = np.sum(x*y) / np.sum(x**2)

    # plt.plot(np.log(mu), - np.log(1 - F), 's')
    #
    # plt.plot(np.log(mu_ok), - np.log(1 - F_ok), 's')
    # plt.plot(np.log(mu), a * np.log(mu))
    #
    # plt.show()
    return a


load_path = "../benchmark_data/manifold_data/"
save_path = "./"
os.makedirs(save_path, exist_ok=True)
datasets = [1, 2, 3, 4, 5, 6, 7, 9, 101, 102, 103, 104, 11, 12, 13]
D = [11, 5, 6, 8, 3, 36, 3, 20, 11, 18, 25, 71, 3, 20, 13]
intdims = [10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]
names = [1, 2, 3, 4, 5, 6, 7,  9, 101, 102, 103, 104, 11, 12, 13]

N = 100
K = 5


result = np.zeros([15, N])


plt.figure()
for j in tqdm(range(15)):
    m = np.zeros(N)
    for i in range(N):
        fn = 'M_{}_{}.mat'.format(datasets[j], i+1)
        M = loadmat(load_path+fn)['x']
        m[i] = compute_2nn(M.T)
    result[j, :] =  m.copy()

np.save(save_path+'2nn_benchmark_res', result)

MPE = compute_mpe(result, np.array([intdims]).T)
MPE2 = compute_mpe(np.round(result), np.array([intdims]).T)

PE = compute_pe(result, np.array([intdims]).T)
print(MPE, MPE2)
print(np.abs(PE).mean(axis=1))
print(np.mean(np.abs(PE)))
print(result.mean(axis=1))

# plt.figure()
plt.boxplot(result.T)
plt.plot(range(1, 16), intdims, 'r_')
#
# # plt.figure()
# # plt.plot(intdims, result, 'bo')
#
plt.show()



