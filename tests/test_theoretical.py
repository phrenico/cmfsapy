from unittest import TestCase
import numpy as np
from cmfsapy.theoretical import theoretical_fsa_pdf


class Test(TestCase):
    def test_theoretical_fsa_pdf(self):
        x = np.arange(0.1, 10, 0.01)
        d = 3
        theoretical_fsa_pdf(x, 1, d)
