import numpy as np
from cmfsapy.dimension.cmfsa import calibrate

n = 2500
realiz_id = 100
my_d = np.arange(2, 81)
myk = 5
box = None

powers = [-1, 1, 2, 3]

coefs = calibrate(n, myk, my_d, realiz_id, powers, box=box)

powers = np.save("powers.npy", powers)
powers = np.save("coefs.npy", coefs)

print(coefs)
# some previous values
#[-6.58771515e-02  2.64684341e-02 -3.47457150e-04  3.81308540e-06]
