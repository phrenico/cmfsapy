from unittest import TestCase
from cmfsapy.dimension.cmfsa import cmfsa
import numpy as np


class Test(TestCase):
    def test_cmfsa(self):
        X = np.random.rand(1000).reshape(100, 10)
        cmfsa(X, 10, [1, 2, 3], [1, 2, 3])
