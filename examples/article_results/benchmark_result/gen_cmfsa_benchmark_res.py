import numpy as np
from scipy.io import loadmat
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

from cmfsapy.dimension.fsa import fsa
from cmfsapy.dimension.correction import correct_estimates


load_path = "../benchmark_data/manifold_data/"
save_path = "/"
os.makedirs(save_path, exist_ok=True)
datasets = [1, 2, 3, 4, 5, 6, 7, 9, 101, 102, 103, 104, 11, 12, 13]
D = [11, 5, 6, 8, 3, 36, 3, 20, 11, 18, 25, 71, 3, 20, 13]
intdims = [10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]
names = [1, 2, 3, 4, 5, 6, 7,  9, 101, 102, 103, 104, 11, 12, 13]

N = 100
K = 5
alphas = np.load('coefs.npy')
powers = np.load('powers.npy')

result = np.zeros([15, N])


plt.figure()
for j in tqdm(range(15)):
    m = np.zeros(N)

    for i in range(1, N+1):
        fn = 'M_{}_{}.mat'.format(datasets[j], i)
        M = loadmat(load_path+fn)['x']

        d = fsa(M.T, k=K)[0][:, -1]
        m[i-1] = np.nanmedian(d)

    result[j, :] = correct_estimates(m.copy(), alpha=alphas, powers=powers)

np.save(save_path+'cmfsa_enchmark_res', result)


# plt.figure()
# plt.boxplot(result.T)
# plt.plot(range(1, 16), intdims, 'r_')
#
# # plt.figure()
# # plt.plot(intdims, result, 'bo')
#
# plt.show()



