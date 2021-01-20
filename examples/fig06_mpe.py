"""Generates Figure6.pdf


To get this, you have to generate test data first,
and also run custom matlab scripts to get the DANCo results first.

0.0. Generate the data
    navigate into the ./benchmark_data/ folder and run the data generating script with GNU octave
        $ octave gen_benchmark_data.m
        
0.1. Generate DANCo results with matlab
@todo:me experiment out hte procedure


1. generates results
2. load DANCo matlab results
3. plot the results

"""
# generate and load data

import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from src.evaluation.mpe import compute_mpe, compute_pe, compute_p_error
from src.algorithm.correction import correct_estimates, compute_mFS_correction_coef
from scipy.io import loadmat
from src.visualization.constants import *



load_path = "../../datasets/processed/ncube/synthetic/"
save_path = "../../reports/"
fn = 'synthetic_res.npy'
matlab_fname = 'danco_dims_M.mat'
big_k_fname = "synthetic_krange20_res.npy"

res = np.load(load_path+fn)
intdims = np.array([[10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]]).T
names = ['$M_1$', '$M_2$', '$M_3$', '$M_4$', '$M_5$', '$M_6$', '$M_7$',  '$M_9$', '$M_{10a}$', '$M_{10b}$',
         r'$M_{10c}$', '$M_{10d}$', '$M_{11}$', '$M_{12}$', '$M_{13}$']
nums = range(1, 16)
instances = 100

# load parameters
K = 5
powers = np.load('powers.npy')
alphas = np.load('coefs.npy')
print("Powers, alphas, k")
print(powers)
print(alphas)
print(K)
print('calibration OK')

# load results and correction
res = np.load(load_path+big_k_fname)[:, :, K-1]
corr_res =  correct_estimates(res, alphas, powers)
corr_res_int = np.round(corr_res)
ml_res =  np.load(load_path+'ml_'+fn)[:, :instances]
danco_res =  np.load(load_path+'danco_'+fn)[:, :instances]
M = loadmat(load_path+matlab_fname)['dims']
danco_matlab_fract = M[:, :, 0]
danco_matlab = M[:, :, 1]
m = res.mean(axis=1)
s = res.std(axis=1)
danco_mat_MPE = compute_mpe(danco_matlab, intdims)
danco_matf_MPE = compute_mpe(danco_matlab_fract, intdims)
levina_MPE = compute_mpe(ml_res, intdims)



mydict = {'dataset':names,
          'd': intdims.flatten(),
          'mFS': m,
          'cmFS frac': corr_res.mean(axis=1),
          'cmFS': corr_res_int.mean(axis=1),
          'DANCo R':np.nanmean(danco_res, axis=1),
          'DANCo M frac':np.nanmean(danco_matlab_fract, axis=1),
          'DANCo M':np.nanmean(danco_matlab, axis=1),
          'Levina': ml_res.mean(axis=1),
          }

MPE = compute_mpe(res, intdims)
cMPE = compute_mpe(corr_res, intdims)
cMPE_int = compute_mpe(corr_res_int, intdims)
danco_MPE = compute_mpe(danco_res, intdims)

print("Szepes median:", MPE.mean())
print("Szepes median:", MPE.mean())
print("Corrected Szepes median:", cMPE.mean())
print("Corrected Szepes median integer:", cMPE_int.mean())

print("DANCo_R:", danco_MPE.mean())
print("DANCo_matlab fractal:", danco_matf_MPE.mean())
print("DANCo_matlab:", danco_mat_MPE.mean())

print("ML:", compute_mpe(ml_res, intdims).mean())


# Plotting the figures
pe = compute_pe(corr_res_int, intdims)
err_mpe = pe.std(axis=1) / np.sqrt(pe.shape[1])


P_err = np.array([compute_p_error(corr_res_int, intdims, axis=1),
                 compute_p_error(danco_matlab, intdims, axis=1)])

print(P_err)


# Plot results
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# subplot 1
x = np.arange(corr_res_int.shape[0])
y = np.array([compute_mpe(corr_res_int, intdims, axis=1),
              compute_mpe(danco_matlab, intdims, axis=1)])
yerr = np.array([compute_pe(corr_res_int, intdims),
                 compute_pe(danco_matlab, intdims)]).std(axis=-1) / 10
barwidth = 0.8/y.shape[0]

_ = [axs[0].bar(x+i*barwidth, Y, width=barwidth, yerr=yerr[i]) for i, Y in enumerate(y)]

axs[0].legend(['cmFSA', 'm-DANCo'])
axs[0].set_ylabel('MPE (%)')

# subplot 2
_ = [axs[1].bar(x+i*barwidth, Y, width=barwidth) for i, Y in enumerate(P_err)]

axs[1].set_xticks(np.arange(0.2, 15.2, 1))
axs[1].set_xticklabels(names)
axs[1].set_ylabel('Error rate')

[axs[i].axes.spines['top'].set_visible(False) for i in range(len(axs))]
[axs[i].axes.spines['right'].set_visible(False) for i in range(len(axs))]
[axs[i].text(-0.08, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(len(axs))]

fig.tight_layout(rect=[0,0,1,1])
# fig.savefig(save_path+"errors.png")
fig.savefig(save_path+"Figure6.pdf", )

print(P_err.mean(axis=1))
plt.show()
