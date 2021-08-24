import numpy as np
from .fsa import fsa
from ..data import gen_ncube
from .correction import correct_estimates, compute_mFSA_correction_coef
from tqdm import tqdm

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

def calibrate(n=2500, myk=20, emb_dims=np.arange(2, 81), N_realiz=100,
              powers=[-1, 1, 2, 3], box=None, load_data=False, save_data=False):
    """Computes regression coefs on calibration dataset with known intrinsic dimensionality

    :param int n: sample size per realization
    :param int myk: maximal neighborhood size
    :param numpy.ndarray of int emb_dims: embedding dimensions to estimate dimension
    :param int N_realiz: number of realizations per embedding dimension
    :param list of in powers: powers to be used in the polynomial basis
    :param None or float box: box size for periodic boundary (default=None)
    :param bool or str load_data: if a string, then took as path to data
    :param bool or str save_data: if a string then took as path to save
    :return: regression coefficients
    :rtype: list of float
    """

    if not load_data:
        # Generate data
        d = _gen_calibration_data(n, emb_dims, myk, N_realiz, box)
        D = np.expand_dims(np.expand_dims(emb_dims, -1), -1)
        k = np.arange(myk + 1)
    else:
        try:
            calibration_res = dict(np.load(load_data+'/calibration_maxk{}_n{}_d{}.npz'.format(myk,
                                                                                              n,
                                                                                              np.max(emb_dims))))
            k = calibration_res['k']
            D = calibration_res['d']
            d = calibration_res['dims']
        except:
            print("Could not load calibration data. \nPlease modify path to 'calibration_maxk{}_n{}_d{}.npz' data file \nor set 'load_data=False' to generate dimension estimates.")

    if save_data:
        _save_calib_data(save_path=save_data, n=n, emb_dims=emb_dims, kmax=myk)
    E = D / d  # relative error of estimates
    coefs = compute_mFSA_correction_coef(d[:, :, myk], E[:, :, myk], powers)

    return coefs

def _gen_calibration_data(n, emb_dims, kmax, N_realiz, box):
    """Generates calibration dataset

    :param int n: sample size per realization
    :param numpy.ndarray of int emb_dims:  embedding dimensions to estimate dimension
    :param int kmax: maximal neighborhood size
    :param int N_realiz: realizations per embedding dimension value
    :return: mFSA dimension estimates with [emb_dims, realizations, k] as axis
    :rtype: numpy.ndarray
    """
    dim_range = []
    for d in tqdm(emb_dims):
        realizations = []
        for j in range(N_realiz):
            X = gen_ncube(n, d, j)
            dims, distances, indices = fsa(X, kmax, boxsize=box)
            realizations.append(dims)
        dim_range.append(realizations)

    dim_range = np.nanmedian(np.array(dim_range), axis=-2)
    return dim_range

def _save_calib_data(save_path, n, d, emb_dims, kmax):
    """Saves out calibration dataset to a path

    :param str save_path: string of data path to save out
    :param numpy.ndarray of float d: mFSA dimension estimates
    :param int n: sample size per realization
    :param numpy.ndarray of int emb_dims:  embedding dimensions to estimate dimension
    :param int kmax: maximal neighborhood size
    :return: None
    """
    np.savez(save_path+"calibration_data_krange{}_n{}_d{}".format(kmax, n, np.max(emb_dims)),
             **{'d':emb_dims.reshape([-1, 1, 1]), 'k':np.arange(myk+1), 'dims': d})