"""This script generates Figure4.pdf

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
overwrite = False  # overwrites the "Figure04_data.pkl" data file if True

Ds = [2, 6, 11, 13, 19, 26]
ns = [10, 100, 500, 1000, 2500, 10000]
colors = ['tab:blue', 'tab:orange', 'tab:green']
realiz_id = 100
fn_data = "Figure04_data.pkl"

if gen_and_measure:
    id_m_dict = {}
    for D in tqdm(Ds):
        id_m = np.array([[np.median(fsa(gen_ncube(n, D, j),
                                        k=1,
                                        boxsize=1)[0][:, 1],
                                    axis=0)
                          for j in range(realiz_id)]
                         for n in tqdm(ns)])
        id_m_dict[D] = id_m
    if overwrite:
        with open(fn_data, 'wb') as f:
            pickle.dump(id_m_dict, f)
else:
    with open(fn_data, 'rb') as f:
        id_m_dict = pickle.load(f)

# Plot results
ylims = [[1, 4],
         [2, 10],
         [5, 15],
         [5, 20],
         [5, 30],
         [10, 40]]

fig, axs = plt.subplots(2, 3, sharex=True, figsize=f26_size)
axs = np.array(axs).flatten()
for i, D in enumerate(Ds):
    axs[i].axhline(D, ls='--', color='k')
    axs[i].plot(ns, id_m_dict[D].mean(axis=1), 's-', color=basic_colors[i])
    axs[i].plot(ns, id_m_dict[D], color=basic_colors[i], alpha=0.05)

    if i in [3, 4, 5]:
        axs[i].set_xlabel(r"n")
    if i in [0, 3]:
        axs[i].set_ylabel(r"$d$")

    axs[i].set_ylim(ylims[i])
    dy = np.ceil((ylims[i][1]-ylims[i][0])/4)
    axs[i].set_yticks(np.arange(ylims[i][0], ylims[i][1]+dy, dy ))
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].grid(True)

axs[0].set_xscale('log')
axs[0].set_xticks([10, 100, 1000, 10000])
[axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(6)]
fig.tight_layout(rect=[0, 0, 1, 1], pad=0, h_pad=0, w_pad=0)
plt.savefig('Figure4.pdf', **save_kwargs)
plt.show()
