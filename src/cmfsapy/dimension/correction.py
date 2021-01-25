import numpy as np

from scipy.odr import Model, RealData, ODR
from functools import partial
# from scipy.stats import norm
# from scipy.linalg import pinv
# from itertools import combinations

# functions
def polynom_func(p, x, powers=[1, 2, 3]):
    """Computes the value of polynomial expression with given mixing coefficients and powers

    :param list of float p: mixing coefficients (length has to match the length of powers)
    :param float or numpy.ndarray of float x: the value of the variable
    :param list of float powers: the powers of (length has to match the length of p)
    :return: the value of the polynomial at the place x
    :rtype: float or numpy.ndarray of float
    """
    return np.array([p[i] * x ** (powers[i]) for i in range(len(powers))]).sum(axis=0)

def compute_mFSA_correction_coef(d, E, powers=[1, 2]):
    """Compute the regression coefficients with Orthogonal Distance Regression

    :param numpy.ndarray of float d: dimension values
    :param numpy.ndarray of float E: relative error
    :param numpy.ndarray of float powers: the powers of the polynomial to include in the regression
    :return: regression coefficients
    :rtype: numpy.ndarray of float
    """
    my_func = partial(polynom_func, powers=powers)
    # Create a model for fitting.
    linear_model = Model(my_func)

    # Create a RealData object using our initiated data from above.
    x = d.mean(axis=1)
    y = np.log(E).mean(axis=1)
    data = RealData(x, y)

    odr = ODR(data, linear_model, beta0=np.random.rand(len(powers)))

    # Run the regression.
    out = odr.run()
    return out.beta


def correct_estimates(d, alpha, powers):
    """Correct mFSA estimates given rergression coefficients and the coresponding powers
    of the polynomial

    :param float d: measured mFSA value(s)
    :param numpy.ndarray of float alpha: regression coefs
    :param numpy.ndarray of float powers: powers of the polynomial
    :return: corrigated-mFSA value(s)
    :rtype: float
    """
    return d * np.exp(polynom_func(alpha, d, powers))


def correct_mFSA(d, E, powers):
    """Correct mFSA values given the relative error of the measurements fit

    :param numpy.ndarray of float d: mFSA values
    :param numpy.ndarray of float E: relative error of mFSA values
    :param numpy.ndarray of float powers:
    :return: corrected estimates
    :rtype: numpy.ndarray of float
    """
    alpha = compute_mFSA_correction_coef(d, E, powers)
    return correct_estimates(d, alpha, powers)