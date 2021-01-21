import numpy as np
from scipy.io import loadmat
import os
from tqdm import tqdm
from statistics import harmonic_mean

import matplotlib.pyplot as plt

from src.algorithm.szepesvari import meausre_dims_fast, ml_dims, szepes_ml


# MPE kiszamitasa: (100/n)∑M(|d^ −d/d)
# de real-world adatnal egy intervallum meajeatol szamitott tavolsagot kell szmolni
load_path = "/home/phrenico/Projects/Codes/dimension-correction/datasets/outer/synthetic/"
save_path = "../../datasets/processed/ncube/synthetic/"
os.makedirs(save_path, exist_ok=True)
datasets = [1, 2, 3, 4, 5, 6, 7, 9, 101, 102, 103, 104, 11, 12, 13]
D = [11, 5, 6, 8, 3, 36, 3, 20, 11, 18, 25, 71, 3, 20, 13]
intdims = [10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]
names = [1, 2, 3, 4, 5, 6, 7,  9, 101, 102, 103, 104, 11, 12, 13]

N = 100
k = 20

result = np.zeros([15, N, k])


for j in tqdm(range(15)):
    m = np.zeros([N, k])
    for i in range(1, N+1):
        fn = 'M_{}_{}.mat'.format(datasets[j], i)
        M = loadmat(load_path+fn)['x']
        d = meausre_dims_fast(M.T, k=k)[0][:, 1:]

        m[i-1] = np.nanmedian(d, axis=0)
    result[j, :] = m.copy()

np.save(save_path+'fsa_krange{}_benchmark_res'.format(k), result)





