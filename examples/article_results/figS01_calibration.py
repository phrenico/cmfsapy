import numpy as np


from scipy.stats import norm
from cmfsapy.dimension.correction import compute_mFSA_correction_coef, correct_estimates, polynom_func
import time
from constants import *

t0 = time.time()


figsave_path = './'

#Load data
calibration_res = dict(np.load('./calibration_result/calibration_data_krange20_n2500_d80.npz'))

k = calibration_res['k']
D = calibration_res['d']
d = calibration_res['dims']
t1 = time.time()
E = D / d
print("k: ", k)

# start correction
K = 5
# powers = [-4, -3, -2, -1, 1, 2, 3]
# powers = [-2, -1, 1, 2]
powers = [-1, 1, 2, 3]
# powers = np.arange(0, 3, 0.5)
# powers = [-1, .5, 1, 2, 3]
# powers = [1]

coefs = compute_mFSA_correction_coef(d[:, :, K], E[:, :, K], powers)
cd =  correct_estimates(d[:, :, K], coefs, powers)
print("coeficients:", coefs)
# np.save('coefs', coefs)
# np.save('powers', powers)
errors = cd-D[:, :, 0]

# computing empirical error probabilities
P_correct = norm.cdf(0.5, loc=errors.mean(axis=1), scale=errors.std(axis=1))\
            - norm.cdf(-0.5, loc=errors.mean(axis=1), scale=errors.std(axis=1))
P_error =  1 - P_correct
P1 = P_error - norm.cdf(-1.5, loc=errors.mean(axis=1), scale=errors.std(axis=1)) \
     - (1- norm.cdf(1.5, loc=errors.mean(axis=1), scale=errors.std(axis=1)))
P2 = P_error - P1- norm.cdf(-2.5, loc=errors.mean(axis=1), scale=errors.std(axis=1)) \
     - (1- norm.cdf(2.5, loc=errors.mean(axis=1), scale=errors.std(axis=1)))
P3 = P_error - P1 - P2



fig = plt.figure(figsize=(9, 8))


plt.subplot(221)
plt.plot(D[:, 0, 0], correct_estimates(d[:, :, K], coefs, powers).round(), 'r.', alpha=0.01, ms=10)
plt.plot(D[:, 0, 0], correct_estimates(d[:, :, K], coefs, powers).round().mean(axis=1), '-', color='gold')
plt.plot(D[:, 0, 0], d[:, :, K], 'b.', alpha=0.01, ms=10)
plt.plot(D[:, 0, 0], D[:, 0, 0], 'k--')
plt.xlim([0, 80])
plt.ylim([0, 80])
plt.xlabel(r'$D$')
plt.ylabel(r'$\hat{d}$')


plt.subplot(222)
_ = plt.plot(d[:, :, K], np.log(E[:, :, K]), 'b.', ms=2)
_ = plt.plot(d[:, :, K].mean(axis=1), np.log(E[:, :, K]).mean(axis=1), '.', ms=5, color='gold')
x = np.arange(1, 48).astype(float)
y = x * np.exp(polynom_func(coefs, x, powers))
_ = plt.plot(x, np.log(y/x), 'r-')
plt.xlim([1, 48])
plt.ylim([0, 0.9])
plt.xlabel(r'$\hat{d}$')
plt.ylabel(r'$\log{E}$')


plt.subplot(223)
plt.plot(D[:, :, 0], errors, alpha=0.2, color='tab:orange')
plt.plot(D[:, :, 0], errors.mean(axis=1), alpha=1, label='$\mu_{\mathrm{error}}$')
plt.plot(D[:, :, 0], errors.mean(axis=1)+3 * errors.std(axis=1), alpha=1, color='tab:blue', ls='--', label='$3\sigma_{\mathrm{error}}$' )
plt.plot(D[:, :, 0], errors.mean(axis=1)-3 * errors.std(axis=1), alpha=1, color='tab:blue', ls='--' )
hline_kwargs = dict(color='k', ls = '--')
[plt.axhline(i, **hline_kwargs) for i in np.arange(-3.5, 4, 1)]
plt.ylim([-4, 4])
plt.xlim([0, 80])
plt.xlabel(r"D")
plt.ylabel(r"error")
plt.legend()


plt.subplot(224)
plt.plot(D[:, :, 0], P_correct, 'ro', label='$P_{\mathrm{correct}}$')
plt.plot(D[:, :, 0], P_error, 'bs', label='$P_{\mathrm{error}}$')
plt.plot(D[:, :, 0], P1, '-', label='$P_{\mathrm{error}} (|E|= 1)$')
plt.plot(D[:, :, 0], P2, '--', label='$P_{\mathrm{error}} (|E|= 2)$')
plt.plot(D[:, :, 0], P3, '-.', label='$P_{\mathrm{error}} (|E|\geq 3)$')
plt.axhline(0.5, **hline_kwargs)
plt.axhline(0.997, **hline_kwargs)
plt.axhline(0.95, **hline_kwargs)
plt.axhline(0.05, **hline_kwargs)
plt.axhline(0.003, **hline_kwargs)
plt.axvline(np.where(P_correct<0.997)[0][0]+2, **hline_kwargs)
plt.axvline(np.where(P_correct<0.95)[0][0]+2, **hline_kwargs)
plt.axvline(np.where(P_correct<0.5)[0][0]+2, **hline_kwargs)
plt.legend()
plt.xlim([0, 80])
plt.xlabel(r"D")
plt.ylabel(r"P")


axs = np.array(fig.axes).flatten()
[axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(len(axs))]
plt.tight_layout(rect=[0, 0, 1, 1])

# fig.savefig(figsave_path+'calibration_k{}_n2500.png'.format(K))
# fig.savefig(figsave_path+'calibration_k{}_n2500.pdf'.format(K))
fig.savefig(figsave_path+'FigureS1.pdf'.format(K))



t_end = time.time()

print(t1-t0, "secs of your time for read-in.")
print(t_end-t0, "secs of your time.")