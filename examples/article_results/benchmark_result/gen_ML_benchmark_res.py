import numpy as np
from scipy.io import loadmat
import os
from tqdm import tqdm
from statistics import harmonic_mean

import matplotlib.pyplot as plt

from cmfsapy.dimension.fsa import ml_dims, szepes_ml, fsa


load_path = "../benchmark_data/manifold_data/"
save_path = "./"

datasets = [1, 2, 3, 4, 5, 6, 7, 9, 101, 102, 103, 104, 11, 12, 13]
D = [11, 5, 6, 8, 3, 36, 3, 20, 11, 18, 25, 71, 3, 20, 13]
intdims = [10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]
names = [1, 2, 3, 4, 5, 6, 7,  9, 101, 102, 103, 104, 11, 12, 13]

N = 100
k1 = 5
k2 = 10

result_ml = np.zeros([15, N])

for j in tqdm(range(15)):
    m_ml = np.zeros(N)
    for i in range(1, N+1):
        fn = 'M_{}_{}.mat'.format(datasets[j], i)
        M = loadmat(load_path+fn)['x']

        d = fsa(M.T, k=k1)[0][:, -1]
        d_ml = ml_dims(M.T, k2=k2, k1=k1)[0]
        m_ml[i - 1] = np.mean(d_ml)

        if j==1:
            bins = np.arange(0, 25, 1)
            plt.figure()
            plt.hist(d_ml, bins=bins)
            plt.hist(d, bins=bins)
            plt.show()
            exit()

    result_ml[j, :] = m_ml.copy()

np.save(save_path+'ml_benchmark_res', result_ml)


# plt.figure()
# plt.boxplot(result.T)
# plt.plot(range(1, 16), intdims, 'r_')
#
# # plt.figure()
# # plt.plot(intdims, result, 'bo')
#
# plt.show()



