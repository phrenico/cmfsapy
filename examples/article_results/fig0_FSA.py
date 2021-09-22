'''Script generating Figure0.pdf
        - presents the FSA estimator
        - example estimates on:
            - hypercube data
            - coupled logistic maps

'''
import numpy as np
import matplotlib.pyplot as plt
# ^^^ pyforest auto-imports - don't write above this line
import pyforest

from cmfsapy.dimension.fsa import fsa
from cmfsapy.dimension.cmfsa import calibrate, cmfsa

from cmfsapy.data import gen_ncube
from scipy.spatial import cKDTree

from matplotlib.patches import Circle, Rectangle, ConnectionPatch



def gen_logistics(x0, N, r, B=[]):
    """Genertes coupled logistic maps with periodic boundary conditions on the [0, 1) interval

    :param numpy.ndarray x0: initial conditions
    :param int N: sample size
    :param float r: parameter of logistic maps
    :param B: coupling matrice
    :return: time series
    :rtype: numpy.ndarray
    """
    def gen_next(x, r, B):
        return np.mod(r * x * (1 - x - np.dot(B, x)), 1)
    x = [x0]
    for i in range(N):
        x.append(gen_next(x[-1], r, B))
    return np.array(x)

# 1. generate random uniform hypercube data
n = 1000 # number of samples
D = 2  # intrinsic dimension
k = 30  # maximum neighborhood size

x = gen_ncube(n, D)

# measure dimesnions
dims = fsa(x, k=k, boxsize=1)[0]
d_mfsa = np.nanmedian(dims, axis=0)
d_efsa = np.nanmean(dims, axis=0)

# 2. Generating logistic map dat
logist_N = 1000  # sample size
B = 0.3 * np.array([[0, 0, 0],[0, 0, 0], [1, 1, 0]])  # coupling matrix
np.random.seed(321)
x0 = np.random.rand(3) # Initial condition

lx = gen_logistics(x0, logist_N, r=3.99, B=B)
LY = np.array([lx[4:, -1], lx[3:-1, -1], lx[2:-2, -1], lx[1:-3, -1]]).T
LX = np.array([lx[2:, 0], lx[1:-1, 0], lx[:-2, 0]]).T

# MEasuring dimensions
lk = 50
lxdims = fsa(LX, k=lk, boxsize=1)[0]
lydims = fsa(LY, k=lk, boxsize=1)[0]

lx_mfsa = np.nanmedian(lxdims, axis=0)
lx_efsa = np.nanmean(lxdims, axis=0)

ly_mfsa = np.nanmedian(lydims, axis=0)
ly_efsa = np.nanmean(lydims, axis=0)


# 3. Demonstraiton plot data
# the random dataset
tree = cKDTree(x)

_, p0_ind = tree.query(np.array([0.5,0.5]), k=1)
p0 = x[p0_ind]

k1 = 15
k2 = 2 *k1
dists, inds = tree.query(p0, k=k2+1)

p1 = x[inds[k1]]
p2 = x[inds[k2]]
I = x[inds]

# 4. Visualizations
circle_kwargs = dict(fill=False, ls='--')
line_kwargs = dict(color='k', ls='-', lw=0.5)

# figure layout
#
# fig2 = plt.figure(figsize=(5,7))
# ax3 = fig2.add_axes([0.5, 0, 1, 1])
# ax2 = fig2.add_axes([0, 0.7,.5,.3])
# ax1 = fig2.add_axes([0.5, 0.7,.5,.3])
# ax0 = fig2.add_axes([0, 0, 0.5, 0.7])

fig2, ax = plt.subplots(3, 2, figsize=(8, 8))
ax2, __, ___, _, aax, bax = np.array(ax).flatten()
ax0 = fig2.add_axes([0.53, 0.53, 0.46, 0.46])
__.remove()
___.remove()
aax.remove()
bax.remove()
ax2.remove()
ax2 = fig2.add_axes([0.08, 0.67, 0.35, 0.33])
ax3 = fig2.add_subplot(312)
ax4 = fig2.add_subplot(313)


def plot_neighborhood(ax):
    ax.plot(*x.T, '.', color='gray')
    ax.plot(*I.T, 'b.')

    ax.plot(*np.array([p0, p1]).T, **line_kwargs)
    ax.plot(*np.array([p0, p2]).T, **line_kwargs)

    ax.plot(*p0, 'r.', ms=10)
    ax.plot(*p1, 'r.', ms=10)
    ax.plot(*p2, 'r.', ms=10)

    circ1 = Circle(p0, np.sqrt(np.sum((p1 - p0) ** 2)), **circle_kwargs)
    circ2 = Circle(p0, np.sqrt(np.sum((p2 - p0) ** 2)), **circle_kwargs)

    ax.add_patch(circ1)
    ax.add_patch(circ2)


def plot_histo(dims, d_mfsa, d_efsa, k1, ax1):
    ax1.hist(dims[:, k1], bins=np.arange(0, 10, 0.15), density=True)

    ax1.plot([d_mfsa[k1]], [0], 'rs')
    ax1.axvline([d_mfsa[k1]], color='r')

    ax1.plot([d_efsa[k1]], [0], 's', color='tab:orange')
    ax1.axvline([d_efsa[k1]], color='tab:orange')
    ax1.axvline(D, ls='--', color='k')
    ax1.text(0.7, 0.8, r"$k={}$".format(k1), transform=ax1.transAxes)


plot_neighborhood(ax0)
plot_neighborhood(ax2)
ax0.set_xlim(0.37, 0.63)
ax0.set_ylim(0.37, 0.63)

ax0.text(0.5, 0.45, r"$R_k$", transform=ax0.transAxes)
ax0.text(0.6, 0.6, r"$R_{2k}$", transform=ax0.transAxes)

ax0.text(0.05, 0.08, r"$\delta_k(x_i) = \frac{\log(2)}{\log(R_{2k}/R_{k})}$", transform=ax0.transAxes, size=16,
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.1', alpha=0.8))

ax3line_kwargs = dict(ls='-', marker='s')


def plot_dimk(dims, D, k, ax):
    d_mfsa = np.nanmedian(dims, axis=0)
    d_efsa = np.nanmean(dims, axis=0)

    ax.plot(range(k + 1), dims.T, '.', color='b', alpha=0.01)
    ax.plot(100, 100, 'o', color='b', alpha=0.1, label=r'$\delta_k$')
    ax.plot(range(k + 1), d_mfsa, **ax3line_kwargs, color='r', label="median")
    ax.plot(range(k + 1), d_efsa, **ax3line_kwargs, color='tab:orange', label="mean")
    ax.axhline(D, color='k', ls='--')
    ax.legend()


plot_dimk(dims, D, k=k, ax=ax3)

plot_dimk(lydims, D=3, k=lk, ax=ax4)
_.remove()

# ax10=fig2.add_subplot(4,4,11)
# ax11=fig2.add_subplot(4,4,12)
# ax12=fig2.add_subplot(4,4,15)
# ax13=fig2.add_subplot(4,4,16)
# mini_ax = [ax10, ax11, ax12, ax13]

# plot_histo(dims, d_mfsa, d_efsa, 1, ax10)
# plot_histo(dims, d_mfsa, d_efsa, 2, ax11)
# plot_histo(dims, d_mfsa, d_efsa, 15, ax12)
# plot_histo(dims, d_mfsa, d_efsa, 30, ax13)

# [i.set_xlim(0, 6) for i in mini_ax]


ax3.set_ylim(0, 7)
ax3.set_xlim(0, 16)
ax3.set_xscale('linear')
ax3.set_xlabel('neighborhood size $k$')
ax3.set_ylabel('')

ax4.set_ylim(0, 10)
ax4.set_xlim(0, 16)
ax4.set_xscale('linear')
ax4.set_xlabel('neighborhood size $k$')
ax4.set_ylabel('')

ax2.axis(False)
ax0.set_xticks([])
ax0.set_yticks([])
xxx = ax0.get_xlim()
yyy = ax0.get_ylim()
print(xxx)

rec = Rectangle([xxx[0], yyy[0]], np.diff(xxx), np.diff(yyy), fill=False, color='k')
ax2.add_patch(rec)
ax3.set_xlabel('')
ax3.set_ylabel('estimated dimension')
ax4.set_ylabel('estimated dimension')

fig2.tight_layout(pad=1, h_pad=2, w_pad=0)
abcd_style = dict(weight='bold', size=18)
ax2.text(-0.03, 2.2, "A", transform=ax3.transAxes, **abcd_style)
ax0.text(0.01, 0.99, "B", transform=ax0.transAxes, va='top', ha='left', **abcd_style)
ax3.text(-0.03, 1.05, "C", transform=ax3.transAxes, **abcd_style)
ax4.text(-0.03, 1.05, "D", transform=ax4.transAxes, **abcd_style)

ax0.set_zorder(100)
con1 = ConnectionPatch(xyA=[xxx[0], yyy[0] + np.diff(yyy)], xyB=[0, 1], coordsA="data", coordsB=ax0.transAxes,
                       axesA=ax2, axesB=ax0, lw=0.5)
con2 = ConnectionPatch(xyA=[xxx[0], yyy[0]], xyB=[0, 0], coordsA="data", coordsB=ax0.transAxes,
                       axesA=ax2, axesB=ax0, lw=0.5)
con3 = ConnectionPatch(xyA=[xxx[0] + np.diff(xxx), yyy[0]], xyB=[1, 0], coordsA="data", coordsB=ax0.transAxes,
                       axesA=ax2, axesB=ax0, lw=0.5)
con4 = ConnectionPatch(xyA=[xxx[0] + np.diff(xxx), yyy[0] + np.diff(yyy)], xyB=[1, 1], coordsA="data",
                       coordsB=ax0.transAxes,
                       axesA=ax2, axesB=ax0, lw=0.5)
fig2.add_artist(con1)
fig2.add_artist(con2)
fig2.add_artist(con3)
fig2.add_artist(con4)


def meanfunc(k, D, a=0.68490195447):
    return D * (a / (k - 1) + 1)


ks = np.arange(2, k, 0.1)
ax3.plot(ks, meanfunc(ks, D=D), '--', color='gray')
# ax4.plot(ks, meanfunc(ks, D=3), '--', color='gray')

# plt.show()
fig2.savefig("Figure0.pdf")