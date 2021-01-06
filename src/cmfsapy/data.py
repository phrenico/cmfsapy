import numpy as np


def gen_ncube(n, d, realization_id=0):
    """Generate reproducible pseudo-random datasets given n, d, and id.

    :param int n: number of data points
    :param int d: dimensionality of the data
    :param int realization_id: which realization you want from the
    :return: n x d numpy array dataset
    """
    np.random.seed(n * d + realization_id)
    return np.random.rand(n*d).reshape([n, d])