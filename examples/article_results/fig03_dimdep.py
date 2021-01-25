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

Ds = np.arange(2, 31)
ns = [10, 100, 500, 1000, 2500, 10000]
colors = ['tab:blue', 'tab:orange', 'tab:green']
realiz_id = 100
fn_data = "Figure03_data.pkl"

if gen_and_measure:
    id_m_dict = {}
    for n in tqdm(ns):
        id_m = np.array([ [np.median(fsa(gen_ncube(n, D, j),
                                   k=1,
                                   boxsize=1)[0][:, 1],
                               axis=0)
                     for j in range(realiz_id)]
                for D in tqdm(Ds)])
        id_m_dict[n] = id_m

    with open(fn_data, 'wb') as f:
        pickle.dump(id_m_dict, f)
else:
    with open(fn_data, 'rb') as f:
        id_m_dict = pickle.load(f)



# Plot the results
fig, axes = plt.subplots(2, 3, sharey=True, sharex=True, figsize=f26_size)
axs = np.array(axes).flatten()
for l in tqdm(range(len(ns))):
    n = ns[l]
    m = id_m_dict[n].mean(axis=1)
    std = id_m_dict[n].std(axis=1)

    axs[l].plot(Ds, Ds, 'k--')
    axs[l].plot(Ds, id_m_dict[n], '-', alpha=0.05, color=basic_colors[l])
    axs[l].plot(Ds, m, '-', color=basic_colors[l], lw=2., label=r"$n={}$".format(n))
    # axs[l].plot(my_d, m - 2*std, '--', color=basic_colors[l], lw=2., label=r"$2 \sigma$")
    # axs[l].plot(my_d, m + 2*std, '--', color=basic_colors[l], lw=2.)
    # axs[l].legend(handlelength=0., loc=2)
    axs[l].set_xlim([2, 30])
    axs[l].set_ylim([2, 30])
    axs[l].grid(True)
    # axs[l].set_title(r"$n={}$".format(n))
    axs[l].text(4, 26, r"$n={}$".format(n), bbox=dict(facecolor='none',
                                                      edgecolor='none',
                                                      alpha=1))
[axs[i].set_xlabel(r'$D$') for i in [3, 4, 5]]
[axs[i].set_ylabel(r'$d$') for i in [0, 3]]
[axs[i].spines['top'].set_visible(False) for i in range(6)]
[axs[i].spines['right'].set_visible(False) for i in range(6)]
[axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(6)]

plt.tight_layout(rect=[0, 0, 1, 1], pad=0, h_pad=0, w_pad=0)

plt.savefig('Figure3.pdf', **save_kwargs)
plt.show()