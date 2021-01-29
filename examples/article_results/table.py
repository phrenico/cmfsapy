"""Genenrates the benchmark table with MPE values

"""

import numpy as np
import pandas as pd

from cmfsapy.evaluation.mpe import compute_mpe
from scipy.io import loadmat
from constants import *

def insert_charachter(mylist, charachter='&'):
    """ inserts a charachter element between each list element

    :param mylist:
    :param charachter:
    :return:
    """
    n_insertion = len(mylist)-2
    n = len(mylist)
    new_list = []
    for i  in range(n):
        for j in range(n_insertion):
            w = i+j
            if (w % 2) == 0:
                new_list.append(mylist[i])
            elif (w % 2) == 1:
                new_list.append(charachter)
    return new_list


load_path = "benchmark_result/"
save_path = "./"
save_fname = 'table_benchmark.tex'

corrected_fn = "cmfsa_benchmark_res.npy"
matlab_fname = 'danco_matlab_benchmark_res.mat'
big_k_fname = "fsa_krange20_benchmark_res.npy"


intdims = np.array([[10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]]).T
names = ['$M_1$', '$M_2$', '$M_3$', '$M_4$', '$M_5$', '$M_6$', '$M_7$',  '$M_9$', '$M_{10a}$', '$M_{10b}$',
         '$M_{10c}$', '$M_{10d}$', '$M_{11}$', '$M_{12}$', '$M_{13}$']
nums = range(1, 16)
instances = 100


K = 5
powers = np.load(load_path+'powers.npy')
alphas = np.load(load_path+'coefs.npy')


res = np.load(load_path+big_k_fname)[:, :, K-1]
corr_res = np.load(load_path+'cmfsa_benchmark_res.npy')
corr_res_int = np.round(corr_res)
ml_res =  np.load(load_path+'ml_benchmark_res.npy')[:, :instances]
danco_res =  np.load(load_path+'danco_r_benchmark_res.npy')[:, :instances]
M = loadmat(load_path+matlab_fname)['dims']
danco_matlab_fract = M[:, :, 0]
danco_matlab = M[:, :, 1]



print("resfile shapes")
print("dancoR: ", danco_res.shape)
print('mFSL:', res.shape)
print('cmFS frac:', corr_res.shape)
print('levina:', ml_res.shape)

m = res.mean(axis=1)
s = res.std(axis=1)

mydict = {'dataset':names,
          'd': intdims.flatten(),
          'mFSA': m,
          'cmFSA frac': corr_res.mean(axis=1),
          'cmFSA': corr_res_int.mean(axis=1),
          'DANCo R':np.nanmean(danco_res, axis=1),
          'DANCo M frac':np.nanmean(danco_matlab_fract, axis=1),
          'DANCo M':np.nanmean(danco_matlab, axis=1),
          'Levina': ml_res.mean(axis=1),
          }

my_df = pd.DataFrame(mydict, index=nums).round(decimals=2)
print(my_df.head())


mystr = my_df.to_latex(save_path+save_fname,
                        escape=False,
                        column_format='llrrrrrrrr',
                        multicolumn_format='c')


MPE = compute_mpe(res, intdims)
cMPE = compute_mpe(corr_res, intdims)
cMPE_int = compute_mpe(corr_res_int, intdims)
danco_MPE = compute_mpe(danco_res, intdims)
danco_mat_MPE = compute_mpe(danco_matlab, intdims)
danco_matf_MPE = compute_mpe(danco_matlab_fract, intdims)
levina_MPE = compute_mpe(ml_res, intdims)


print("Szepes median:", MPE.mean())
print("Szepes median:", MPE.mean())
print("Corrected Szepes median:", cMPE.mean())
print("Corrected Szepes median integer:", cMPE_int.mean())

print("DANCo_R:", danco_MPE.mean())
print("DANCo_matlab fractal:", danco_matf_MPE.mean())
print("DANCo_matlab:", danco_mat_MPE.mean())

print("ML:", compute_mpe(ml_res, intdims).mean())

my_str = '&MPE& & {:.2f} & {:.2f}& {:.2f}& {:.2f}&{:.2f}&{:.2f}&{:.2f}\n'.format(MPE.mean(),
                                                                             cMPE.mean(),
                                                                             cMPE_int.mean(),
                                                                             danco_MPE.mean(),
                                                                             danco_matf_MPE.mean(),
                                                                             danco_mat_MPE.mean(),
                                                                             levina_MPE.mean())
with open(save_path+save_fname, 'r') as f:
    table = f.readlines()
new_contents = [table[i] for i in range(len(table)-1)] + [my_str] + [table[-1]]
table.insert(-1, my_str)


with open(save_path+save_fname, 'w') as f:
    table = "".join(table)
    f.write(table)