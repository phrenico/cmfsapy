import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from sympy.plotting import plot_implicit

from constants import *

import matplotlib.colors as mcolors
import bz2


def plot_grid(grid, ax, is_colbar=False, cm_range=None, cm='inferno', cb_txt=''):
    ylabels = ['Gr-{}'.format(abc[0])] + ['{}'.format(i) for i in abc[1:6]] + ['Fb-{}'.format(abc[0])] +['{}'.format(i) for i in abc[1:2]] + ['JIH', 'BIH', 'JT']

    if cm_range is None:
        cm_range = dict(vmin=grid.min(), vmax=grid.max())
    im = ax.imshow(grid,
                   aspect='auto',
                   cmap=cm,
                    **cm_range)

    ax.set_xticks(range(8))
    ax.set_xticklabels(range(1, 9))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(ylabels)
    hline_col = 'k'
    hline_pos = [5.5, 7.5, 8.5, 9.5]
    [ax.axhline(i, color=hline_col, lw=1) for i in hline_pos]

    if is_colbar:
        plt.colorbar(im, ax=ax, label=cb_txt)
    else:
        return im


def plot_median_grid_duo(cgrid, sgrid, titles=['control', 'seizure'], cm='PiYG'):
    args = [cgrid, sgrid]
    cm_norm1 = dict(norm=mcolors.DivergingNorm(vcenter=np.max(cgrid)))
    cm_norm2 = dict(norm=mcolors.DivergingNorm(vcenter=np.max(sgrid)))
    cm_norm = [cm_norm1, cm_norm2]
    cm = [cm, cm + "_r"]
    fig, axs = plt.subplots(1, 2, sharey=True)
    ims = [plot_grid(np.median(args[i], axis=0),
                     axs[i], cm_range=cm_norm[i], cm=cm[i], is_colbar=True) for i in range(len(args))]
    _ = [axs[i].set_title(title) for i, title in enumerate(titles)]

    _ = [axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(2)]
    fig.tight_layout(rect=[0, 0, 1, 1], pad=0, h_pad=0, w_pad=1)

    # cb_ax = fig.add_axes([0.85, 0.05, 0.05, 0.9])
    # cb_ax.set_ylabel(r'$d$')
    # fig.colorbar(ims[0], cax=cb_ax)
    return fig

def plot_median_grid_trio(cgrid, sgrid, titles=['control', 'seizure'], cm='plasma', cm_diff='PiYG'):
    args = [cgrid, sgrid]
    cmap_max = max([np.max(cgrid), np.max(sgrid)])
    cmap_min = max([np.min(cgrid), np.min(sgrid)])

    cm_norm1 = dict(norm=mcolors.Normalize(vmin=cmap_min, vmax=cmap_max))
    cm_norm2 = cm_norm1  # dict(norm=mcolors.DivergingNorm(vmin=cmap_min, vmax=cmap_max))
    cm_norm = [cm_norm1, cm_norm2]
    cm = [cm, cm]
    fig, axs = plt.subplots(2, 3, sharey='row', figsize=(12, 9))
    axs = [axs[1, i] for i in range(3)]
    ims = [plot_grid(np.median(args[i], axis=0),
                     axs[i], cm_range=cm_norm[i], cm=cm[i], is_colbar=True, cb_txt='') for i in range(len(args))]
    _ = [axs[i].set_title(title) for i, title in enumerate(titles)]

    axs[2].set_title("seizure-control")
    norm = dict(norm=mcolors.DivergingNorm(vcenter=0))
    im = plot_grid(np.median(sgrid, axis=0) - np.median(cgrid, axis=0),
                   axs[2], cm_range=norm, cm=cm_diff, cb_txt=r"$\Delta d$")
    fig.colorbar(im, ax=axs[2])


    _ = [axs[i].text(-0.15, 1.01, tagging[i+2], transform=axs[i].transAxes, **tag_kwargs) for i in range(3)]
    fig.tight_layout(rect=[0, 0, 1, 1], pad=0, h_pad=0, w_pad=1)

    # cb_ax = fig.add_axes([0.85, 0.05, 0.05, 0.9])
    # cb_ax.set_ylabel(r'$d$')
    # fig.colorbar(ims[0], cax=cb_ax)
    return fig

def plot_grid_many(grid, shape=(4, 5), titles=['']):
    fig, axs = plt.subplots(*shape, sharex=True, sharey=True, figsize=f45_size)
    axs = np.array(axs).flatten()
    ims = [plot_grid(grid[i],
                     axs[i],
                     cm_range=dict(vmin=np.min(grid),vmax=np.max(grid)))
           for i in range(grid.shape[0])]

    _ = [axs[i].set_title('{}'.format(titles[i]*2048), fontsize=7) for i in range(len(titles))]
    axs[-1].set_xticks(range(0, 8, 2))
    axs[-1].set_xticklabels(range(1, 9, 2))

    ax0 = fig.add_axes([0.85, 0.05, 0.05, .9])
    cbar = fig.colorbar(ims[0], cax=ax0)

    fig.tight_layout(rect=[0, 0, 0.85, 1], pad=1, h_pad=0.5, w_pad=1)
    return fig


def plot_boxplot(grid, ax):
    shapes = grid.shape
    x = grid.reshape([shapes[0], -1])

    hm = ax.boxplot(x)
    return hm

def plot_boxplot_duo(cgrid, sgrid, titles=['control', 'seizure']):
    fig, axs = plt.subplots(2, 1, sharey=True, sharex=True, figsize=fbig_size)
    _ = plot_boxplot(cgrid, axs[0])
    _ = plot_boxplot(sgrid, axs[1])
    [ax.set_title(titles[i]) for i,ax in enumerate(axs)]
    axs[1].set_xticklabels([])
    axs[1].set_xlabel('recording-channels')
    [axs[i].set_ylabel(r'$\hat{d}$') for i in range(len(axs))]
    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


if __name__=="__main__":
    # load_file = "avg_Figure07_data.pkl"
    load_file = "./epi_data/Figure07_data.pkl"


    save_path = "./"
    res_dict = pd.read_pickle(load_file)
    channels = res_dict['channels']
    abc = string.ascii_uppercase

    ckeys, cgrid = res_dict['control']
    skeys, sgrid = res_dict['seizure']



    # plot Figure 7
    fig = plot_median_grid_trio(cgrid, sgrid, cm='jet_r', cm_diff='bwr_r')
    axs = fig.get_axes()
    _ = [axs[i].axis('off') for i in range(3)]

    ypad_s = 0.07
    ypad_t = 0.1
    xpad_s = 0
    xpad_r = 0.025
    axa = fig.add_axes([0. , 0.5 + ypad_s, 0.5 - xpad_r, 0.5 - ypad_t])
    axe = fig.add_axes([0.5 + xpad_s, 0.5 + ypad_s, 0.5 - xpad_r, 0.5 - ypad_t])
    axa.text(-0.15, 2.1, tagging[0], transform=axs[3].transAxes, **tag_kwargs)
    axe.text(1.8, 2.1, tagging[1], transform=axs[3].transAxes, **tag_kwargs)
    axa.axis('off')

    print(os.listdir('./'))
    im = plt.imread('./MRI_Lateral3.png')
    axa.imshow(im)

    path = './epi_data/avg_dims_dict.pkl'
    with bz2.BZ2File(path, 'rb') as f:
        dimdata = pickle.load(f)

    path2 = './epi_data/dims_dict.pkl'
    with bz2.BZ2File(path2, 'rb') as f:
        mdimdata = pickle.load(f)

    instance = 0
    case = 'seizure'

    times = list(dimdata[case].keys())

    e_mdimdata = np.array([np.array(mdimdata[case][i]).mean(axis=0) for i in times])
    e_dimdata = np.array([np.array(dimdata[case][i]).mean(axis=0) for i in times])

    ax = axe
    kcols = ['tab']

    for k in range(3, e_mdimdata.shape[-1]):
        ax.plot(e_mdimdata.mean(axis=0)[:, k - 1], e_dimdata.mean(axis=0)[:, k - 1], '.', label=r'k={}'.format(k))
    b = 10
    ax.set_xlim(2, b)
    ax.set_ylim(2, b)
    ax.plot(range(0, b + 1), 'k-', lw=0.5)
    ax.legend()
    ax.set_xlabel(r'$d_{\mathrm{mFSA}}$')
    ax.set_ylabel(r'$d_{\mathrm{FSA}}$')

    ax2 = fig.add_axes([0.75, .605, .2, .2])

    _ = ax2.plot(range(1, 14), e_mdimdata[instance].T, 'r', alpha=0.03)
    _ = ax2.plot(range(1, 14), e_dimdata[instance].T, 'tab:orange', alpha=0.03)
    _ = ax2.plot(range(1, 14), e_dimdata[instance].T.mean(axis=1), 'tab:orange', label='FSA', lw=2)
    _ = ax2.plot(range(1, 14), e_mdimdata[instance].T.mean(axis=1), 'red', label='mFSA', lw=2)

    ax2.set_ylim(2, 10)
    ax2.set_xlim(0, 14)
    ax2.legend()
    ax2.set_ylabel('id')
    ax2.set_xlabel(r'$k$', labelpad=-7)

    axs[3].set_ylabel("electrode row")
    axs[3].set_xlabel("electrode column")
    axs[4].set_xlabel("electrode column")
    axs[5].set_xlabel("electrode column")
    # axs[4].set_ylabel("electrode row4")
    # axs[5].set_ylabel("electrode row5")

    fig.tight_layout(rect=[0, 0, 1, 1], pad=1, h_pad=2, w_pad=1)
    axs[3].text(1.08, -0.05, r"$d$", transform=axs[3].transAxes)
    axs[4].text(1.08, -0.05, r"$d$", transform=axs[4].transAxes)
    axs[5].text(1.07, -0.05, r"$\Delta d$", transform=axs[5].transAxes)


    fig.savefig(save_path+'Figure7.pdf', **save_kwargs)
    # fig.savefig(save_path+'Figure7.png', **save_kwargs)

    # plt.show()

