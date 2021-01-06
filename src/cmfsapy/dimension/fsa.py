import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import cpu_count

def get_dists_inds_ck(X, k, boxsize):
    tree = cKDTree(X, boxsize=boxsize)
    dists, inds = tree.query(X, k + 1, n_jobs=cpu_count())
    return dists, inds

def szepesvari_dimensionality(dists):
    """Compute szepesvari dimensions from kNN distances

    :param dists:
    :return:
    """
    n = dists.shape[1]
    lower_k = np.arange(np.ceil(n / 2)).astype(int)
    upper_k = np.arange(n)[::2]
    d = - np.log(2) / np.log(dists[:, lower_k] / dists[:, upper_k])
    return d

def fsa(X, k, boxsize=None):
    """Measure local Szepesvari-Farahmand dimension, distances are computed by the cKDTree algoritm

    :param arraylike X: data series [n x dim] shape
    :param k: maximum k value
    :param boxsize: apply d-toroidal distance computation with edge-size =boxsize, see ckdtree class for more
    :return: local estimates, distances, indicees
    """
    dists, inds = get_dists_inds_ck(X, 2*k, boxsize)
    dims = szepesvari_dimensionality(dists)
    return dims, dists, inds