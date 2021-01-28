import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import cpu_count
from scipy.stats import hmean


def get_dists_inds_ck(X, k, boxsize):
    """computes the kNN distances and indices

    :param numpy.ndarray X:  2D array with data shape: (ndata, n_vars)
    :param int k: neighborhood size
    :param float boxsize:  circular boundary condition to [0, boxsice] interval for all input dimensions if not None.
    :return: KNN distances and indices
    """
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

def ml_estimator(normed_dists):
    return -1./ np.nanmean(np.log(normed_dists), axis=1)

def ml_dims(X, k2, k1=1):
    """Maximum likelihood estimator af intrinsic dimension (Levina-Bickel)"""
    dists, inds = get_dists_inds_ck(X, k2+1, boxsize=None)
    norm_dists = dists / dists[:, -1:]
    dims = ml_estimator(norm_dists[:, k1:-1])
    return dims, dists, inds

def szepes_ml(local_d):
    """maximum likelihood estimator from local FSA estimates (for k=1)

    :param numpy.ndarray of float local_d: local FSA estimates
    :return: global ML-FSA estimate
    """
    return  hmean(local_d) / np.log(2)