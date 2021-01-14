import numpy as np
from .fsa import fsa
from .correction import correct_estimates

def cmfsa(X, k, powers=None, alphas=None, boxsize=None):
    """Computes corrigated dimension estimations on dataset
    @todo: Hard-wire default values from the article

    :param numpy.ndarray of float X: data
    :param int k:
    :param list of float powers: powers
    :param list of float alphas: regression coeffitients
    :param float boxsize:
    :return:
    """
    fsa_dims = np.nanmedian(fsa(X, k, boxsize)[0], axis=0)
    cmfsa_dims = correct_estimates(fsa_dims, alphas, powers)

    return cmfsa_dims