import numpy as np

from scipy.odr import Model, RealData, ODR
from functools import partial
# from scipy.stats import norm
# from scipy.linalg import pinv
# from itertools import combinations

# functions
def polynom_func(p, x, powers=[1, 2, 3]):
    return np.array([p[i] * x ** (powers[i]) for i in range(len(powers))]).sum(axis=0)

def compute_mFS_correction_coef(d, E, powers=[1, 2]):
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
    return d * np.exp(polynom_func(alpha, d, powers))


def correct_mFS(d, E, powers):
    alpha = compute_mFS_correction_coef(d, E, powers)
    return correct_estimates(d, alpha, powers)