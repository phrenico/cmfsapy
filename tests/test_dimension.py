from unittest import TestCase
import numpy as np
from cmfsapy.dimension import fsa

class Test(TestCase):
    def test_fsa(self):
        X = np.random.rand(1000).reshape(100, 10)
        dims, dists, inds = fsa(X, k=19, boxsize=1)