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

f, axs = plt.subplots(2, 2, figsize=(fbig_size[0], 1.3 * fbig_size[1]))
axs = axs.flatten()
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
[axs[i].set_xlabel(r'$d$') for i in range(n_ax)]
axs[0].legend(handlelength=0.6)
axs[0].set_xlim([0, 5])
axs[1].set_xlim([0, 10])
axs[0].set_ylim([0, 5])
axs[1].set_ylim([0, 2])
axs[1].set_yticks([0, 0.8, 1.6])
[axs[i].grid(True) for i in range(n_ax)]
[axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(n_ax)]
axs[0].set_ylabel('density')

# Draw last 2 subplots too
def get_mean(x, p):
    return np.nansum(x * p * np.diff(x)[0])


def get_variance(x, p):
    dx = np.diff(x)[0]
    return np.nansum(x ** 2 * p * dx) - get_mean(x, p) ** 2

# here is the sample size dependence

x = np.arange(0, 50, 0.001)
k = 1
Ns = np.logspace(2, 3, num=10)

K = []
ds = [2, 5, 12]
sdevmeds = []
markers = ['s', 'd', '.']
cols = ['tab:blue', 'tab:orange', 'tab:green']

for d_ind, d in enumerate(ds):
    fu = np.array([unnormed_median_df(x, k, N, d) for N in Ns]).T
    v_med = np.array([get_variance(x, fu[:, i]) for i in range(fu.shape[1])]).T
    sdev_med = np.sqrt(v_med)
    sdevmeds.append(sdev_med)

    coefs = np.polyfit(np.log(Ns), np.log(sdev_med), deg=1)
    y = np.polyval(coefs, np.log(Ns))
    K.append(coefs)

    axs[2].plot(Ns, np.exp(y), label=r'$D={}$'.format(d), color='{}'.format(cols[d_ind]))

    axs[2].plot(Ns, sdev_med, color='{}'.format(cols[d_ind]),
                marker='{}'.format(markers[d_ind]), lw=0, ms=8)

    axs[2].plot(Ns, sdev_med / d, color='{}'.format(cols[d_ind]),
                marker='{}'.format(markers[d_ind]), lw=0, ms=8)


axs[2].plot(Ns, sdev_med / d, 'k--', lw=2,
            label=r'$\mathrm{err}/D$')
K = np.array(K)

axs[2].legend(handlelength=0.6, loc='upper right')

axs[2].set_xscale('log')
axs[2].set_yscale('log')

axs[2].set_xlabel(r"$n$ (sample size)", labelpad=1)
axs[2].set_ylabel(r"standard error", labelpad=-10)
axs[2].set_ylim([0, None])


# And here is the neighborhood dependence
x = np.arange(0, 50, 0.001)
ks = np.arange(1, 200, 10)
N = 1e3
d = 2

K = []
ds = [2, 5, 12]
sdevmeds = []
markers = ['s', 'd', '.']
cols = ['tab:blue', 'tab:orange', 'tab:green']
# inset = f.add_axes([0.82, 0.27, 0.15, 0.15])
# axs[2].cla()
# inset.cla()
for d_ind, d in enumerate(ds):
    fu = np.array([unnormed_median_df(x, k, N, d) for k in ks]).T
    v_med = np.array([get_variance(x, fu[:, i]) for i in range(fu.shape[1])]).T
    sdev_med = np.sqrt(v_med)
    sdevmeds.append(sdev_med)

    coefs = np.polyfit(np.log(ks), np.log(sdev_med / d), deg=1)
    y = np.polyval(coefs, np.log(ks))
    print(coefs, y.shape)
    K.append(coefs)

    axs[3].plot(ks, d*np.exp(y), label=r'$D={}$'.format(d), color='{}'.format(cols[d_ind]))

    axs[3].plot(ks, sdev_med, color='{}'.format(cols[d_ind]),
                marker='{}'.format(markers[d_ind]), lw=0, ms=8)

    axs[3].plot(ks, sdev_med / d, color='{}'.format(cols[d_ind]),
                marker='{}'.format(markers[d_ind]), lw=0, ms=8)

axs[3].plot(ks, sdev_med / d, 'k--', lw=2,
            label=r'$\mathrm{err}/D$')

# axs[3].legend(handlelength=0.6, loc='upper right')

axs[3].set_xscale('log')
axs[3].set_yscale('log')

bbox_kwarg = dict(facecolor='none',
                  edgecolor='none',
                  alpha=1)
text_kwarg = dict(horizontalalignment='left',
                  verticalalignment='bottom')
axs[3].set_xlabel(r"$k$ (neighborhood size)", labelpad=1)
# axs[3].set_ylabel(r"standard error", labelpad=-10)
axs[3].text(0.05, 0.05, r'$n={}$'.format(int(N)),
            transform=axs[3].transAxes, bbox=bbox_kwarg, **text_kwarg)
axs[2].text(0.05, 0.05, r'$k=1$',
            transform=axs[2].transAxes, bbox=bbox_kwarg, **text_kwarg)



K = np.array(K)
sdevmeds = np.array(sdevmeds).T



f.tight_layout(rect=[0.01, 0, 1, 1], pad=0.5, h_pad=0, w_pad=1)


f.savefig("Figure2.pdf", **save_kwargs)



# f.tight_layout(rect=[0.01, 0, 1, 1], pad=0, h_pad=0, w_pad=1)
# f.savefig("Figure2.pdf", **save_kwargs)

# plt.show()
