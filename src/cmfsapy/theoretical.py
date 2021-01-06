import numpy as np
from scipy.special import comb

def theoretical_szepes_pdf(w, k, d):
    """Analitic pdf for Szepesvari-Farahmand dimension estimator

    :param numpy.ndarray w: the value of the estimator
    :param int k: kNN
    :param int d: intrinsic dimensionality
    :return: pdf value at w
    :rtype: numpy.ndarray
    """
    return (2*k-1)*comb(2*k-2,k-1)*d*(np.log(2)/w**2) * np.exp(-np.log(2)*d*k/w)*(1 - np.exp(- np.log(2)*d / w))**(k-1)