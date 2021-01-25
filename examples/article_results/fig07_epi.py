import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from sympy.plotting import plot_implicit

from constants import *

import matplotlib.colors as mcolors


def plot_grid(grid, ax, is_colbar=False, cm_range=None, cm='inferno'):
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
        plt.colorbar(im, ax=ax)
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
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 5))
    ims = [plot_grid(np.median(args[i], axis=0),
                     axs[i], cm_range=cm_norm[i], cm=cm[i], is_colbar=True) for i in range(len(args))]
    _ = [axs[i].set_title(title) for i, title in enumerate(titles)]

    axs[2].set_title("seizure-control")
    norm = dict(norm=mcolors.DivergingNorm(vcenter=0))
    im = plot_grid(np.median(sgrid, axis=0) - np.median(cgrid, axis=0),
                   axs[2], cm_range=norm, cm=cm_diff)
    fig.colorbar(im, ax=axs[2])


    _ = [axs[i].text(-0.15, 1.01, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(3)]
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
    load_file = "Figure07_data.pkl"
    save_path = "../"
    res_dict = pd.read_pickle(load_file)
    channels = res_dict['channels']
    # print(res_dict.keys())
    abc = string.ascii_uppercase

    ckeys, cgrid = res_dict['control']
    skeys, sgrid = res_dict['seizure']

    # print(sgrid.shape)
    # print(cgrid.shape)
    # print(channels)
    # print(string.ascii_lowercase[0])

    my_cmap = 'bwr'
    # # save median figures()
    # fig = plot_median_grid_duo(cgrid, sgrid, cm=my_cmap)
    # fig.savefig(save_path+'06-_grid.pdf', **save_kwargs)

    # save median figures()
    fig = plot_median_grid_trio(cgrid, sgrid, cm='jet_r', cm_diff='bwr_r')
    fig.savefig(save_path+'Figure7.pdf', **save_kwargs)
    # plt.show()

    # # save median differences figure
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title("seizure-control")
    # norm = dict(norm=mcolors.DivergingNorm(vcenter=0))
    # im = plot_grid(np.median(sgrid, axis=0)-np.median(cgrid, axis=0),
    #                     ax, cm_range=norm, cm=my_cmap+'_r')
    # fig.colorbar(im)
    # fig.savefig(save_path+"07-_grid_diff.pdf")
    #
    # #save many little subplots
    # fig = plot_grid_many(cgrid, titles=ckeys)
    # fig.savefig(save_path+'S-_control_grid.pdf', **save_kwargs)
    #
    # fig = plot_grid_many(sgrid, titles=skeys)
    # fig.savefig(save_path+'S-_seizure_grid.pdf', **save_kwargs)
    #
    #
    # #save boxplot
    # fig = plot_boxplot_duo(cgrid, sgrid)
    # fig.savefig(save_path+'S-boxplots.pdf', **save_kwargs)

