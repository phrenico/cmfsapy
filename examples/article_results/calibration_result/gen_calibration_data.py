import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cmfsapy.dimension.fsa import fsa
from cmfsapy.data import gen_ncube

import os


save_path = "./"


ns = [2500]

colors = ['tab:blue', 'tab:orange', 'tab:green']
realiz_id = 100
my_d = np.arange(2, 81)
myk = 20
box = None


for l, n in enumerate(ns):
    dim_range = []
    for d in tqdm(my_d):
        realizations = []
        for j in range(realiz_id):
            X = gen_ncube(n, d, j)
            dims, distances, indices = fsa(X, myk, boxsize=box)
            realizations.append(dims)
        dim_range.append(realizations)

    dim_range = np.nanmedian(np.array(dim_range), axis=-2)

    np.savez(save_path+"calibration_data_krange{}_n{}_d{}".format(myk, n, d), **{'d':my_d.reshape([-1, 1, 1]),
                                                                            'k':np.arange(myk+1),
                                                                            'dims': dim_range})
    print(dim_range.shape)
