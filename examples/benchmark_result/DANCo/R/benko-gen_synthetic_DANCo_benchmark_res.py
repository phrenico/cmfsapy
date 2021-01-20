#!coding=utf-8
# This is a script for python2 !!!
# rpy2 is not compatible with python3
import numpy as np
from scipy.io import loadmat
import os
from tqdm import tqdm

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
idr = importr('intrinsicDimension')
rpy2.robjects.numpy2ri.activate()



load_path = "../../../benchmark_data/manifold_data/"
save_path = "../../"
# os.makedirs(save_path, exist_ok=True)

datasets = [1, 2, 3, 4, 5, 6, 7, 9, 101, 102, 103, 104, 11, 12, 13]
D = [11, 5, 6, 8, 3, 36, 3, 20, 11, 18, 25, 71, 3, 20, 13]
intdims = [10, 3, 4, 4, 2, 6, 2, 20, 10, 17, 24, 70, 2, 20, 1]
names = [1, 2, 3, 4, 5, 6, 7,  9, 101, 102, 103, 104, 11, 12, 13]
fns = ['M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'M_7',  'M_9', 'M_101', 'M_102',
         'M_103', 'M_104', 'M_11', 'M_12', 'M_13']


caldata = idr.DancoCalibrationData(k=10, N=2500)

for i in tqdm(range(80)):
    caldata =  idr.increaseMaxDimByOne(caldata)


iterations = 100
res = np.zeros([15, iterations])
for j in tqdm(range(15)):
    for i in tqdm(range(1, iterations+1), leave=False):
        fn = '{}_{}.mat'.format(fns[j], i)
        x = loadmat(load_path+fn)['x'].T[:2500, :]
        xr = ro.r.matrix(x, nrow=x.shape[0], ncol=x.shape[1])

        try:
            myd = idr.dancoDimEst(xr, 10, 80, **{'calibration.data':caldata})
            res[j, i-1] = float(myd[0][0])
        except:
            print("Sg went wrong", j, i)
            myd = np.nan
            res[j, i-1] = myd

np.save(save_path+'danco_synthetic_res', res)

import matplotlib.pyplot as plt
# plt.figure()
# plt.boxplot(result.T)
# plt.plot(range(1, 16), intdims, 'r_')
#
# # plt.figure()
# # plt.plot(intdims, result, 'bo')
#
# plt.show()



