import numpy as np

from constants import *





load_path = "./epi_embed/"

with open(load_path+'embedding_results', 'rb') as f:
    # results = pickle.load(f)
    in_dict = np.load(f, allow_pickle=True)
    print(list(in_dict.keys()))
    locals().update(in_dict)  # define variables from dictionary


k1 = 10
k2 = 20

fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(8, 8))
axs = np.array(axs).flatten()
for i in range(12):
    _ = axs[i].plot(control_res_dict[ i +1].T)
    _ = axs[i].set_yticks(np.arange(0, 12, 2))
    _ = axs[i].set_ylim([0, 9])


    _ = axs[i].set_xlim([0, 30])
    _ = axs[i].grid(True)
    _ = axs[i].set_title("D={}".format( i +1))
    _ = axs[i].axvline(k1, ls='--')
    _ = axs[i].axvline(k2, ls='--')
    if i>= 9:
        _ = axs[i].set_xlabel(r'$k$')
    if i in [0, 3, 6, 9]:
        _ = axs[i].set_ylabel(r'$\hat{d}$')

pad = 2
wpad = 1
hpad = 1

fig.tight_layout(rect=[0, 0, 1, 0.7], pad=pad, h_pad=hpad, w_pad=wpad)

pad = 0.1
wpad = 0.1
hpad = 0.05
ax = fig.add_axes([0. + pad, 0.7 + hpad, 0.5 - wpad / 2 - pad, 0.3 - pad])
ax1 = fig.add_axes([0.5 + wpad / 2, 0.7 + hpad, 0.5 - wpad / 2 - pad, 0.3 - pad])

alphas = 0.1
_ = ax.plot(deltas / 2.048, sst, color="tab:green", alpha=alphas)
_ = ax.plot(deltas / 2.048, sst25, color="tab:orange", alpha=alphas)
_ = ax.plot(deltas / 2.048, sst1, color="tab:blue", alpha=alphas)

_ = ax.plot(deltas / 2.048, sst.mean(axis=1), color="tab:green", lw=3, label=r'$50 \%$')
_ = ax.plot(deltas / 2.048, sst25.mean(axis=1), color="tab:orange", lw=3, label=r'$25 \%$')
_ = ax.plot(deltas / 2.048, sst1.mean(axis=1), color="tab:blue", lw=3, label=r'$1 \%$')

ax.legend(loc='lower right')
# ax.set_xlim([0, np.max(deltas)/2.048])
ax.set_xlim([1, 50])

ax.set_xlabel(r'$\Delta t $ (ms)')
ax.set_ylabel(r'$\Delta x$ (au)')
ax.set_yscale('log')
plt.grid(True)

_ = ax1.plot(Ds, emb_dims)
_ = ax1.plot(Ds, Ds, 'k--')
_ = ax1.axhline(7, ls='--')

ax1.set_ylabel(r'$\hat{d}$')
ax1.set_xlabel(r'$D$')
_ = axs[0].text(-0.15, 1.15, tagging[2], transform=axs[0].transAxes, **tag_kwargs)
_ = ax.text(-0.15, 1.05, tagging[0], transform=ax.transAxes, **tag_kwargs)
_ = ax1.text(-0.15, 1.05, tagging[1], transform=ax1.transAxes, **tag_kwargs)
save_path = "./"
fig.savefig(save_path + 'FigureS2.pdf')
# plt.show()