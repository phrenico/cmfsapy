"""Generate Figure 1 from the ground up

"""

import numpy as np
import matplotlib.pyplot as plt

from cmfsapy.data import gen_ncube
from cmfsapy.dimension.fsa import fsa
from cmfsapy.theoretical import theoretical_fsa_pdf

from constants import f26_size, tagging, tag_kwargs, save_kwargs


def stringadder(a, b):
    """Helper function to formulate string addition over lists (for plotting)

    :param list of str a: a string description of the param
    :param list of int b: an integer value
    :return: list of joined strings
    :rtype: list of str
    """
    return [a[i]+str(b[i]) for i in range(len(a))]

if __name__=="__main__":
    # Parameters
    n_sample = 10000
    Ds =  [2, 3, 5, 8, 10, 12]
    ks = [1, 11, 50]

    dw = 0.001
    w = np.arange(0, 200, dw)

    # Generate data, measure dimensions, compute theoretical densities
    dims_dict = {}
    theor_dict = {}
    for D in Ds:
        # Generate data
        x = gen_ncube(n_sample, D, 0)

        # Measure local dimensions
        d = fsa(x, max(ks), boxsize=1)[0][:, ks]
        dims_dict[D] = d

        # compute theoretical pdf of values
        dims_theor = np.array([theoretical_fsa_pdf(w, i, D) for i in ks]).T
        theor_dict[D] = dims_theor



    # Plot the results
    f, axs = plt.subplots(2, 3, figsize=f26_size, sharey='row')
    axs = np.array(axs).flatten()
    bins = np.arange(-0.5, 100, 0.1)

    for cycler, D in enumerate(Ds):
        myax = axs[cycler]
        dims_theor = theor_dict[D]
        d = dims_dict[D]


        myax.plot(w, dims_theor, linewidth=3.)

        myax.set_prop_cycle(None)
        for i, myk in enumerate(ks):
            _ = myax.hist(d[:, i], density=True, bins=bins, alpha=0.5)

        myax.set_xlim([0, max([12, D + 5])])

        myax.grid(True)
        if D == 2:
            myax.legend(stringadder(['k=', 'k=', 'k='], np.array(ks)))

    # f.suptitle('Theory vs. Simulations on Random uniforn n-cube \n $n$={}, periodic boundary'.format(n))
    _ = [ax.set_xlabel(r'$\delta$') for ax in axs[[3, 4, 5]]]

    for i in range(len(axs)):
        axs[i].set_title(r'$D={}$'.format(Ds[i]))
        [axs[i].text(-0.15, 1.05, tagging[i], transform=axs[i].transAxes, **tag_kwargs) for i in range(len(axs))]
    f.tight_layout(rect=[0, 0, 1, 1], pad=1, h_pad=0, w_pad=0)
    f.savefig('Figure1.pdf', **save_kwargs)
    plt.show()




