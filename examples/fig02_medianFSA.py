import numpy as np
from cmfsapy.dimension.fsa import fsa
from cmfsapy.data import gen_ncube
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np

from scipy.special import betainc, beta
from scipy.stats import beta as bet
from scipy.integrate import quad, simps

import matplotlib.pyplot as plt

from cmfsapy.theoretical import theoretical_fsa_pdf
from constants import *


def unnormed_median_df(w, k, N, d):
    m = N//2
    x = 2**(-d/w)
    p = betainc(k, k, x)
    dx_dw = d * np.log(2) * 2**(-d/w) / w**2
    return  (p * (1 - p)) ** (m) * bet(k, k).pdf(x) * dx_dw / beta(m+1, m+1)

def unnormed_1k_median(w, N, d):
    p = 2**(-d/w)
    dx_dw = d * np.log(2) * 2 ** (-d / w) / w ** 2
    return (p*(1-p))**(N/2) #* dx_dw

def median_pdf(w, f=unnormed_median_df, **kwargs):
    p = f(w, **kwargs)
    p = p #/ simps(p, w)
    return p

# Generate dataset and measure dimension
# plot results


# Generate
ns = [11, 101, 1001]
n_realization = 5000
myk = 1

print("Generate data...")
dataset = {}
for n in tqdm(ns):
    save_dict = {}
    for D in tqdm([2, 5]):
        dims_array = []
        for realiz_id in range(n_realization):
                X = gen_ncube(n, D, realiz_id)
                dims, distances, indices = fsa(X, myk, boxsize=1)

                dims = np.median(dims, axis=0)
                dims_array.append(np.array(dims))

        dims_array = np.array(dims_array)[:, 1:]
        save_dict[D] = dims_array
    dataset[n] = save_dict

# Plot data
print("Plot results...")
dw = 0.01
w = np.arange(0.01, 100, dw)
t = np.arange(0, 100, 5*dw)
ns = [11, 101, 1001]

f, axs = plt.subplots(1, 2, figsize=(f26_size[0], f26_size[1]/2))
for i in tqdm(range(3)):
    id_m = dataset[ns[i]]


    axs[0].plot(w, median_pdf(w, k=1, N=ns[i], d=2), color=basic_colors[i], label="$n={}$".format(ns[i]))
    axs[1].plot(w, median_pdf(w, k=1, N=ns[i], d=5), color=basic_colors[i])

    axs[0].hist(id_m[2], density=True, bins=t, color=basic_colors[i], alpha=0.5)
    axs[1].hist(id_m[5], density=True, bins=t, color=basic_colors[i], alpha=0.5)

    if i ==0:
        bbox_kwarg = dict(facecolor='none',
                          edgecolor='none',
                          alpha=1)
        text_kwarg = dict(horizontalalignment='left',
                          verticalalignment='top')
        axs[0].text(0.1, 4.9, r"$D=2$", bbox=bbox_kwarg, **text_kwarg)
        axs[1].text(0.2, 2-0.04, r"$D=5$", bbox=bbox_kwarg, **text_kwarg)

n_ax = len(axs)
[axs[i].set_xlabel(r'$m_d$') for i in range(n_ax)]
axs[0].legend(handlelength=0.7)
axs[0].set_xlim([0, 5])
axs[1].set_xlim([0, 10])
axs[0].set_ylim([0, 5])
axs[1].set_ylim([0, 2])
axs[1].set_yticks([0, 0.8, 1.6])
[axs[i].grid(True) for i in range(n_ax)]
[axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(n_ax)]

f.tight_layout(rect=[0.01, 0, 1, 1], pad=0, h_pad=0, w_pad=1)
f.savefig("Figure2.pdf", **save_kwargs)

plt.show()