"""This script generates Figure3.pdf

1. Generate data and measure intrinsic dimensions
2. Visualization

"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from cmfsapy.data import gen_ncube
from cmfsapy.dimension.fsa import fsa

from constants import *


# Generate data and measure dimensions for different embedding dimensions (D), sample_sizes (n), 100 realization each
# The dimensions computed with periodic boundary condition in the unit n-cube
gen_and_measure = False  # if True, then generates dimension measurements and save it into Figure03_data.pkl (It takes hours)
overwrite = False  # overwrites previously generated data in "Figure05_data.pkl" if True

Ds = np.arange(2, 31)
ns = [10, 100, 500, 1000, 2500, 10000]
colors = ['tab:blue', 'tab:orange', 'tab:green']
realiz_id = 100
fn_data = "Figure05_data.pkl"

if gen_and_measure:
    id_m_dict = {}
    for n in tqdm(ns):
        id_m = np.array([ [np.median(fsa(gen_ncube(n, D, j),
                                   k=1,
                                   boxsize=None)[0][:, 1],
                               axis=0)
                     for j in range(realiz_id)]
                for D in tqdm(Ds)])
        id_m_dict[n] = id_m
    if overwrite:
        with open(fn_data, 'wb') as f:
            pickle.dump(id_m_dict, f)
else:
    with open(fn_data, 'rb') as f:
        id_m_dict = pickle.load(f)



# plot data
def corr(d, alpha):
    return d*np.exp(alpha*d)

fig, axes = plt.subplots(2, 3, sharey=True, sharex=True, figsize=f26_size)
axs = np.array(axes).flatten()
alphas = []
for l in tqdm(range(len(ns))):
    n = ns[l]
    id_m = id_m_dict[n]

    Y = np.log(Ds).reshape([-1, 1])
    Z = np.log(id_m)
    deltaD = Y - Z

    a = np.sum(deltaD * id_m) / np.sum(id_m**2)

    alphas.append(a)
    if ns[l] == 2500:
        print("2500-nal az alpha:", a)

    axs[l].plot(Ds, Ds, 'k--')

    axs[l].plot(Ds, id_m, '-', alpha=0.01, color='grey')
    axs[l].plot(Ds, corr(id_m, a).mean(axis=-1), '-', color=basic_colors[l])

    axs[l].plot(Ds, id_m.mean(axis=-1), '-', color='grey')
    axs[l].plot(Ds, corr(id_m, a), '-', color=basic_colors[l], alpha=0.01)

    axs[l].text(4, 26, r"$n={}$".format(n), bbox=dict(facecolor='none',
                                                      edgecolor='none',
                                                      alpha=1))

    axs[l].text(4, 23, r"$\alpha={:.3f}$".format(a), bbox=dict(facecolor='none',
                                                               edgecolor='none',
                                                               alpha=1))

axs[l].set_xlim([min(Ds), max(Ds)])
axs[l].set_ylim([min(Ds), max(Ds)])
[axs[i].grid(True) for i in range(6)]
[axs[i].set_xlabel(r'$D$') for i in [3, 4, 5]]
[axs[i].set_ylabel(r'$d$') for i in [0, 3]]
[axs[i].spines['top'].set_visible(False) for i in range(6)]
[axs[i].spines['right'].set_visible(False) for i in range(6)]
[axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(6)]

# axs[0].set_yscale('log')
# axs[0].set_xscale('log')

plt.tight_layout(rect=[0, 0, 1, 1], pad=0, h_pad=0, w_pad=0)

plt.savefig('Figure5.pdf', **save_kwargs)

plt.show()