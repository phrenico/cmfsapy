"""Generate the intrinsic dimension estimates for the epileptic recordings

1. Load data chunks
2. Compute Current Source Densities and measure dimensions
3. Save out results
"""
import numpy as np
import os
import matplotlib.pyplot as plt


import sys
sys.path.append("/home/phrenico/Projects/Codes/dimension-correction/")

import neo
from mpl_toolkits.mplot3d import Axes3D

from mne.filter import filter_data

from cmfsapy.preprocessing.tde import TimeDelayEmbedder
from tqdm import tqdm

from cmfsapy.dimension.fsa import fsa
import pickle
from scipy.linalg import block_diag
from sklearn.preprocessing import scale
import bz2


def sec_2_index(secs, fr=2048, offset=0):
    """

    :param numpy.ndarray secs: time in seconds to be converted
    :param float fr: dampling frequency
    :param float offset: offset in seconds
    :return: time in samples
    :rtype: numpy.ndarray
    """
    return (secs*fr - offset*fr).astype(int)

# def compute_dims_OLD(mychunk, fr, lfreq, hfreq, D, tau, subsample, maxk):
#     """Computes the intrinsic dimensionality
#
#     :param np.ndarray data_chunk: data chunk from measurements [time-instance, dims]
#     :param float fr: sampling frequency of the signal
#     :param float lfreq: lower limit of bandpass filter
#     :param float hfreq: upper limit of bandpass filter
#     :param int D: embedding dimension
#     :param int tau: embedding delay
#     :param int subsample: subsample embeddings by this
#     :param int maxk: max neighborhood size to estimate intrinsic dimension
#     :return: median FS estimate for k values 1-maxk
#     :rtype: np.ndarray of float
#     """
#     n_signals = mychunk.shape[1]
#     x = filter_data(mychunk.T, sfreq=fr,
#                     l_freq=lfreq, h_freq=hfreq, verbose=False).T
#
#     X = TimeDelayEmbedder(d=D, tau=tau,
#                           subsample=subsample).fit_transform(x)
#     # X = np.random.normal(0, 1, size=[10000, 5])
#
#
#     dims = []
#     for i in tqdm(range(n_signals), leave=False):
#         d = meausre_dims_fast(X[:, i, :], k=maxk)[0][:, :]
#         dims.append(np.median(d, axis=0))
#     dims= np.array(dims)
#     return dims

def compute_dims(mychunk, fr, lfreq, hfreq, D, tau, subsample, maxk, subs_method=None):
    """Computes the intrinsic dimensionality

    :param np.ndarray data_chunk: data chunk from measurements [time-instance, dims]
    :param float fr: sampling frequency of the signal
    :param float lfreq: lower limit of bandpass filter
    :param float hfreq: upper limit of bandpass filter
    :param int D: embedding dimension
    :param int tau: embedding delay
    :param int subsample: subsample embeddings by this
    :param int maxk: max neighborhood size to estimate intrinsic dimension
    :return: median FS estimate for k values 1-maxk
    :rtype: np.ndarray of float
    """
    n_signals = mychunk.shape[1]
    x = filter_data(mychunk.T, sfreq=fr,
                    l_freq=lfreq, h_freq=hfreq, verbose=False).T

    # Time-delay embedding
    X = np.array([TimeDelayEmbedder(d=D, tau=tau).fit_transform(x[:, i]) for i in range(x.shape[1])]).transpose([1, 0, 2])
    n0 = X.shape[0]


    # Random subsets
    if subs_method=='random':
        inds = np.random.permutation(np.arange(n0))
        c = n0 // subsample
        a = np.arange(0, n0 + 1, c)

        subsampleds = []
        for i in range(len(a) - 1):
            subsampleds.append(X[inds, :, :][a[i]:a[i + 1], :, :])
    else:
        # Not random subsets
        subsampleds = []
        for i in range(subsample):
            subsampleds.append(X[i::subsample, :, :])

    subs_dims = []
    for Y in subsampleds:
        dims = []
        for i in range(n_signals):
            d = fsa(Y[:, i, :], k=maxk)[0][:, :]
            dims.append(np.nanmedian(d, axis=0))
        dims= np.array(dims)
        subs_dims.append(dims.copy())
    return subs_dims

def get_gridded_from_locals(res_dict, indices, k1, k2):
    grid = np.zeros([len(res_dict.keys()), len(indices) // 8, 8])

    keys = list(res_dict.keys())
    for i, key in enumerate(keys):
        D = np.mean(np.mean(res_dict[key], axis=0)[:, k1:k2], axis=1)
        grid[i, :, :] = D.reshape([len(indices) // 8, 8])
    return keys, grid

if __name__=="__main__":
    # load data
    with bz2.BZ2File('./raw_data_dicts.pckl', 'rb') as f:
        raw_data = pickle.load(f)

    fr = raw_data['samplefreq']
    channels = raw_data['channels']
    control_chunk_dict = raw_data['control']
    seizure_chunk_dict = raw_data['seizure']

    #graph Laplace for CSD computation
    M = np.diag(np.ones(47), k=1) + np.diag(np.ones(47), k=-1) + np.diag(np.ones(40), k=8) + np.diag(np.ones(40), k=-8)
    N = np.diag(np.ones(15), k=1)+ np.diag(np.ones(15), k=-1) + np.diag(np.ones(8), k=8) + np.diag(np.ones(8), k=-8)
    O = np.diag(np.ones(7), k=1)+ np.diag(np.ones(7), k=-1)
    K = block_diag(M, N, O, O, O)  # Adjacency matrix legoed from each small matrix
    Lapl = np.eye(K.shape[0]) * K.sum(axis=1) - K  # graph-laplace to compute CSD (Degree - Adjacency)

    # plt.figure()
    # plt.matshow(L)
    # plt.show()
    # exit()

    # Start the procedure
    indices = list(range(64))  + list(range(65, 89))
    lfreq = 1 # Hz
    hfreq = 30 # Hz
    D = 7
    tau = 17
    subs = 10
    k = 12

    control_res_dict = {}
    for time in tqdm(control_chunk_dict.keys()):
        #Get the chunk
        data_chunk = control_chunk_dict[time][:, indices].astype(float)

        #CSD computation
        data_chunk = scale(data_chunk)
        csd_chunk = scale(np.dot(data_chunk, Lapl))

        #Compute dims
        dims = compute_dims(csd_chunk, fr, lfreq, hfreq, D, tau, subs, k)
        # test_dims = compute_dims(data_chunk, fr, lfreq, hfreq, D, tau, subs, k)
        control_res_dict[time] = dims

    seizure_res_dict = {}
    for time in tqdm(seizure_chunk_dict.keys()):
        data_chunk = seizure_chunk_dict[time][:, indices].astype(float)

        #CSD computation
        data_chunk = scale(data_chunk)
        csd_chunk = scale(np.dot(data_chunk, Lapl))

        #Compute dims
        dims = compute_dims(csd_chunk, fr, lfreq, hfreq, D, tau, subs, k)
        seizure_res_dict[time] = dims

    intermittent_save_path = "./"
    with bz2.BZ2File(intermittent_save_path+'dims_dict.pkl', 'wb') as f:
        pickle.dump({'control': control_res_dict,
                     'seizure': seizure_res_dict,
                     'channels': channels}, f)

    #Select specific channels for visualizations
    k1 = 5
    k2 = 10

    s_keys, seizure_grid = get_gridded_from_locals(seizure_res_dict, indices, k1, k2)
    c_keys, control_grid = get_gridded_from_locals(control_res_dict, indices, k1, k2)

    # print(seizure_grid.mean(axis=0))

    # 3. Save out results for Figure 7
    save_path = "./"
    with open(save_path+"Figure07_data.pkl", 'wb') as f:
        pickle.dump({'control': (c_keys, control_grid),
                     'seizure': (s_keys, seizure_grid),
                     'channels': channels[indices]}, f)