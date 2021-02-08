"""Generates data for the embedding plots

1. Load data
2. Measure intrinsic dimensions in the function of embedding dimension for a seizure
3. compute space-time separation diagrams
4. save out results

"""

import os
import bz2
import numpy as np
from mne.filter import filter_data
from tqdm import tqdm
from scipy.linalg import block_diag
from sklearn.preprocessing import scale
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import pickle
from cmfsapy.preprocessing.tde import TimeDelayEmbedder
import sys
sys.path.append("../")
sys.path.append("../../")
from gen_epi_dimvals import compute_dims, sec_2_index
from constants import *




def shift_data_pair(X, delta):
    return X[:-delta, :], X[delta:, :]

def compute_distance(X, Y, q=2):
    return np.sum((X-Y)**2, axis=-1)**(1/q)

def compute_4_one(i, X, bar=None):
    if bar:
        bar.update(1)
    return compute_distance(*shift_data_pair(X=X, delta=i))

def sts(X, max_delta):
    """Space-Time separation diagram

    :param X: data
    :param max_delta: maximal timehift to look for
    :return: shifts and space time separation values
    """
    deltas = np.arange(1, max_delta+1, 1).astype(int)
    # mybar = tqdm(total=len(deltas), leave=False)
    computer = partial(compute_4_one, X=X, bar=None)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as mypool:
        dx = mypool.map(computer, deltas)
    # mybar.close()
    return deltas, list(dx)

def act_on_list(f):
    def wrapper(my_list, *args, **kwargs):
        res = [f(my_list[i], *args, **kwargs) for i in tqdm(range(len(my_list)),
                                                            leave=False)]
        return res
    return wrapper


#Load data and computations
# load data
with bz2.BZ2File('../raw_data_dicts.pkl', 'rb') as f:
    raw_data = pickle.load(f)

fr = raw_data['samplefreq']
channels = raw_data['channels']
control_chunk_dict = raw_data['control']
seizure_chunk_dict = raw_data['seizure']

data = {**control_chunk_dict, **seizure_chunk_dict}

#graph Laplace for CSD computation
M = np.diag(np.ones(47), k=1) + np.diag(np.ones(47), k=-1) + np.diag(np.ones(40), k=8) + np.diag(np.ones(40), k=-8)
N =  np.diag(np.ones(15), k=1)+ np.diag(np.ones(15), k=-1) + np.diag(np.ones(8), k=8) + np.diag(np.ones(8), k=-8)
O =   np.diag(np.ones(7), k=1)+ np.diag(np.ones(7), k=-1)
K = block_diag(M, N, O, O, O)  # Adjacency matrix legoed from each small matrix
Lapl = np.eye(K.shape[0]) * K.sum(axis=1) - K  # graph-laplace to compute CSD (Degree - Adjacecy)

indices = list(range(64))  + list(range(65, 89))
lfreq = 1 # Hz
hfreq = 30 # Hz
Ds = range(1, 13)
tau = 17
subs = 10
k = 40

# dimension measurements on the first seizure time series
emb_res_dict = {}
for D in tqdm(Ds):
    time = list(seizure_chunk_dict.keys())[0]
    #Get the chunk
    data_chunk = seizure_chunk_dict[time][:, indices]

    #CSD computation
    data_chunk = scale(data_chunk)
    csd_chunk = scale(np.dot(data_chunk, Lapl))

    #Compute dims
    dims = compute_dims(csd_chunk, fr, lfreq, hfreq, D, tau, subs, k)
    # test_dims = compute_dims(data_chunk, fr, lfreq, hfreq, D, tau, subs, k)  # measure dims on LFP instead of CSD
    emb_res_dict[D] = dims

emb_dims =  []
for i in emb_res_dict.keys():
    emb_dims.append(emb_res_dict[i][0][:, 5:10].mean(axis=1))
emb_dims = np.array(emb_dims)


# 2. Space-Time Separation Plots
times_sec_control = list(control_chunk_dict.keys())
times_sec_seizure = list(seizure_chunk_dict.keys())
times_sec_all = np.concatenate([times_sec_control, times_sec_seizure])
q1 = act_on_list(partial(np.percentile, q=1))
q25 = act_on_list(partial(np.percentile, q=25))
q50 = act_on_list(partial(np.percentile, q=50))

sst = []
sst1 = []
sst25 = []
for i in tqdm(range(len(times_sec_all))):
    # read in data
    time = times_sec_all[i]
    print(time)
    # Get the chunk
    data_chunk = data[time]

    # CSD computation
    data_chunk = scale(data_chunk)[:, indices]
    csd_chunk = scale(np.dot(data_chunk, Lapl))

    # filtering
    x = filter_data(csd_chunk.T, sfreq=2048,
                    l_freq=1, h_freq=30, verbose=False).T

    # embed
    X = np.array([TimeDelayEmbedder(d=D, tau=tau).fit_transform(csd_chunk[:, i])
                  for i in range(csd_chunk.shape[1])]).transpose([1, 0, 2])

    # space-time separation
    deltas, dx = sts(X, 120)
    dx1 = q1(dx, axis=0)
    dx25 = q25(dx, axis=0)
    dx50 = q50(dx, axis=0)

    avg1 = np.nanmean(dx1, axis=1)
    avg25 = np.nanmean(dx25, axis=1)
    avg50 = np.nanmean(dx50, axis=1)

    sst.append(avg50)
    sst1.append(avg1)
    sst25.append(avg25)

sst = np.array(sst).T
sst1 = np.array(sst1).T
sst25 = np.array(sst25).T




out_dict = dict(control_res_dict=emb_res_dict,
                deltas=deltas,
                sst=sst,
                sst25=sst25,
                sst1=sst1,
                Ds = Ds,
                emb_dims=emb_dims)

save_path = "./"
os.makedirs(save_path, exist_ok=True)
with open(save_path+'embedding_results', 'wb') as f:
    pickle.dump(out_dict, f)