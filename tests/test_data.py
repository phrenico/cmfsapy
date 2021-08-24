from unittest import TestCase
from cmfsapy.data import gen_ncube

class Test(TestCase):
    def test_gen_ncube(self):
        ds = range(1, 21)
        n = 1000

        [gen_ncube(n, d, 5) for d in ds]
