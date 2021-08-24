from unittest import TestCase
from cmfsapy.dimension.correction import correct_estimates, polynom_func, \
    compute_mFSA_correction_coef, correct_mFSA
import numpy as np


class Test(TestCase):
    def test_correct_estimates(self):
        d = np.random.rand(100)
        alpha = [0.1, 0.2, 0.3]
        powers = [1, 2, 3]
        correct_estimates(d, alpha, powers)

    def test_polynom_func(self):
        self.assertAlmostEqual(polynom_func([1, 2, 3], 2, [1, 2, 3]), 34)

    def test_compute_m_fs_correction_coef(self):
        d = np.random.rand(1000).reshape(100, 10)
        powers = [1, 2, 3]
        E = d / 2
        coefs = compute_mFSA_correction_coef(d, E, powers)


    def test_correct_m_fsa(self):
        d = np.random.rand(1000).reshape(100, 10)
        powers = [1, 2, 3]
        E = d / 2
        cmFSA = correct_mFSA(d, E, powers)
