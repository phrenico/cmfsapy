import numpy as np
from scipy.io import loadmat
import os
from tqdm import tqdm
from statistics import harmonic_mean

import matplotlib.pyplot as plt

from cmfsapy.dimension.fsa import fsa, ml_dims, szepes_ml


# MPE kiszamitasa: (100/n)∑M(|d^ −d/d)
# de real-world adatnal egy intervallum meajeatol szamitott tavolsagot kell szmolni
load_path = "../benchmark_data/manifold_data/"
save_path = "./"

datasets = [1, 2, 3, 4, 5, 6, 7, 9, 101, 102, 103, 104, 11, 12, 13]
D = [11, 5, 6, 8, 3, 36, 3, 20, 11, 18, 25, 71, 3, 20, 13]
intdims = [10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]
names = [1, 2, 3, 4, 5, 6, 7,  9, 101, 102, 103, 104, 11, 12, 13]

N = 100

result = np.zeros([15, N])
result_ml = np.zeros([15, N])

plt.figure()
for j in tqdm(range(15)):
    m = np.zeros(N)
    m_ml = np.zeros(N)
    for i in range(1, N+1):
        fn = 'M_{}_{}.mat'.format(datasets[j], i)
        M = loadmat(load_path+fn)['x']


        d = fsa(M.T, k=1)[0][:, -1]
        d_ml = ml_dims(M.T, k2=20, k1=6)[0]
        m[i-1] = np.nanmedian(d)
        m_ml[i - 1] = szepes_ml(d)  # harmonic_mean(d_ml)


    result[j, :] = m.copy()
    result_ml[j, :] = m_ml.copy()

# np.save(save_path+'synthetic_res', result)
# np.save(save_path+'ml_synthetic_res', result_ml)


# plt.figure()
# plt.boxplot(result.T)
# plt.plot(range(1, 16), intdims, 'r_')
#
# # plt.figure()
# # plt.plot(intdims, result, 'bo')
#
# plt.show()



