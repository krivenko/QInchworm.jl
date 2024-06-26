

import glob

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import h5py

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 8

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#plt.rcParams["mathtext.fontset"] = "cm"
#plt.rc('mathtext', fontset="cm")
#plt.rc('text', usetex=True)


filenames = [
    #'data_analytic_chain_integral_epsilon_0.0_beta_1.0_V_1.0_n_max_16.h5',
    #'data_analytic_chain_integral_epsilon_0.25_beta_1.0_V_1.0_n_max_16.h5',
    #'data_analytic_chain_integral_epsilon_0.5_beta_1.0_V_1.0_n_max_16.h5',
    #'data_analytic_chain_integral_epsilon_1.0_beta_1.0_V_1.0_n_max_16.h5',
    #'data_analytic_chain_integral_epsilon_2.0_beta_1.0_V_1.0_n_max_16.h5',
    #"data_analytic_chain_integral_beta_1_V_1_n_max_128_invlap_linear.h5",
    #'data_analytic_chain_integral_beta_1_V_1_n_max_13_invlap.h5',
    'data_analytic_chain_integral_beta_2_V_1_n_max_13_invlap_log.h5',
    'data_analytic_chain_integral_beta_1_V_1_n_max_13_invlap_log.h5',
    'data_analytic_chain_integral_beta_0.5_V_1_n_max_13_invlap_log.h5',
    'data_analytic_chain_integral_beta_0.25_V_1_n_max_13_invlap_log.h5',
    'data_analytic_chain_integral_beta_0.0_V_1_n_max_13_invlap_log.h5',
    ]

gs = GridSpec(
    1, 1,
    width_ratios=[1],
    height_ratios=[1],
    wspace=0.0, hspace=0.45,
    bottom=0.25, top=0.99,
    left=0.13, right=1.04,
    )

fig = plt.figure(figsize=(3.25, 1.25))
ax = plt.subplot(gs[0])

for filename in filenames:
    print(f'--> Loading: {filename}')
    with h5py.File(filename, 'r') as f:
        f = f['data']
        beta = f.attrs['beta']
        epsilon = f.attrs['epsilon']
        V = f.attrs['V']
        Ds = np.array(f['Ds'])
        ns = np.array(f['ns'])

    be = sp.nsimplify(beta * epsilon)
    l = plt.plot(ns, Ds, '.', label=rf'$\beta \epsilon = {sp.latex(be)}$', alpha=0.75, lw=1.0)
    color = l[0].get_color()
    plt.plot(ns, Ds, '-', alpha=0.25, lw=1.5, color=color)

    #ns = np.arange(8, 17)
    #ns = np.array([100, 1500])
    ns = np.array([8, 1024*8])
    plt.plot(ns, 0*ns + np.exp(-3/2 * epsilon * beta), '--', color=color, lw=1.0, alpha=0.75)

plt.plot([], [], '--', color='gray', lw=1.0, alpha=0.75, label=r'$e^{-3\beta\epsilon/2}$')
plt.xlabel(r'$n$', labelpad=1)
plt.ylabel(r'$d_n$', labelpad=1)
#plt.grid(True)
plt.ylim(bottom=0)
#plt.xlim(left=0)
#plt.xticks(range(0, 16 + 4, 4))
plt.semilogx([], [])

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.43))

plt.savefig('figure_dn.pdf')
#plt.show()

fig = plt.figure(figsize=(3.25, 3.25))

for filename in filenames:
    print(f'--> Loading: {filename}')
    with h5py.File(filename, 'r') as f:
        f = f['data']
        beta = f.attrs['beta']
        epsilon = f.attrs['epsilon']
        V = f.attrs['V']
        Ds = np.array(f['Ds'])
        ns = np.array(f['ns'])

    be = sp.nsimplify(beta * epsilon)

    dscale = Ds - np.exp(-3/2 * epsilon * beta)
    
    l = plt.plot(ns, dscale, '.', label=rf'$\beta \epsilon = {sp.latex(be)}$', alpha=0.75, lw=1.0)
    color = l[0].get_color()
    plt.plot(ns, dscale, '-', alpha=0.25, lw=1.5, color=color)

    ns = np.arange(8, 17)
    #plt.plot(ns, 0*ns + np.exp(-3/2 * epsilon * beta), '--', color=color, lw=1.0, alpha=0.75)


#plt.semilogy([], [])
plt.loglog([], [])
plt.legend()
    
plt.tight_layout()
    
plt.savefig('figure_dn_scale.pdf')
    
