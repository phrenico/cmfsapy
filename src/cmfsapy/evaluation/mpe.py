import numpy as np

def compute_mpe(estimates, d, axis=None):
    """Computes the  Mean Percentage Error

    :param np.ndarray estimates: estimated values
    :param np.ndarray d: ground-truth
    :return: MPE score
    :rtype: float, np.ndarray
    """
    MPE = 100 * np.nanmean((np.abs(estimates-d) / d), axis=axis)
    return MPE

def compute_pe(estimates, d):
    """Computes the  Percentage Error

    :param np.ndarray estimates: estimated values
    :param np.ndarray d: ground-truth
    :return: MPE score
    :rtype: float, np.ndarray
    """
    PE = 100 * (estimates-d) / d
    return PE

def compute_p_error(x, intdims, axis=None):
    """Computes the error rate for integer dimension estimates (1-hit rate)

    :param int x: the intrinsic dimension of the manifold
    :param numpy.ndarray of int intdims: id estimates
    :param axis: axis to compute the mean (default None)
    :return: error rate
    :rtype: float or numpy.ndarray of float
    """
    p_err = 1 - (x==intdims).mean(axis=axis)
    return p_err