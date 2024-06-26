

import glob

import numpy as np
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


class Dummy():
    def __init__(self): pass

def load_h5(filename):
    
    with h5py.File(filename, 'r') as fd:
        g = fd['data']
        #print(g.keys())
        #print(g.attrs.keys())        

        d = Dummy()
        d.parm = Dummy()
        for key, val in g.attrs.items():
            setattr(d.parm, key, val)
        for key, val in g.items():
            setattr(d, key, np.array(val))

    return d

path = '../../../experiments/diagrammatics/'
#filename = '../../experiments/diagrammatics/data_analytic_chain_integral.h5'
#filename = path + 'data_analytic_chain_integral_beta_10_V_20_n_max_128.h5'
#filename = path + 'data_analytic_chain_integral_beta_1000_V_1.375_n_max_256.h5'
#filename = path + 'data_analytic_chain_integral_beta_1_V_100.0_n_max_128.h5'
#filename = path + 'data_analytic_chain_integral_beta_1_V_1_n_max_128.h5'
#filename = path + 'data_analytic_chain_integral_beta_1_V_1_n_max_256.h5'
#filename = path + 'data_analytic_chain_integral_beta_1_V_1_n_max_512.h5'
#filename = path + 'data_analytic_chain_integral_beta_1_V_1_n_max_1024.h5'
filename = path + 'data_analytic_chain_integral_beta_1_V_1_n_max_13_invlap.h5'

print(f'--> Loading: {filename}')
ref = load_h5(filename)

filenames = np.sort(glob.glob('data_*h5'))

results = []
for filename in filenames:    
    print(f'--> Loading: {filename}')
    r = load_h5(filename)
    results.append(r)

# -- order by order convergence

plt.figure(figsize=(3.25, 2.0))

gs = GridSpec(
    1, 1,
    width_ratios=[1],
    height_ratios=[1],
    wspace=0.0, hspace=0.45,
    bottom=0.18, top=0.99,
    left=0.16, right=0.99,
    )

plt.subplot(gs[0])

print(ref.ns)
print(ref.Ds)
print(ref.ns.dtype)

#exit()

n = 22
    
for r in sorted(results, key=lambda r: r.parm.order):

    if r.parm.order < 4: continue
    
    #if len(ref.Ds) < r.parm.order:
    #    exact = ref.Ds[-1]
    #else:
    #    exact = ref.Ds[r.parm.order - 1]

    if r.parm.order in ref.ns:
        idx = list(ref.ns).index(r.parm.order)
        print(idx, ref.Ds[idx])
        exact = ref.Ds[idx]

    print(f"order = {r.parm.order:4d} val = {exact}")
    #exact = np.exp(-3/2)
    
    rel_err = np.abs((r.results - exact) / exact)
    plt.plot(r.N_samples_list[:n], rel_err[:n], '.-', label=f'{r.parm.order}', alpha=0.5)

n = np.logspace(1.0, 8.75)
plt.plot(n, 15/n, '--k', lw=0.5, alpha=1.0, label='$\sim N^{-1}$')
    
plt.loglog([], [])
plt.legend(title='Diagram order', ncol=1, fontsize=6, loc='center right')
plt.grid(True)
plt.axis('equal')
plt.xlabel(r'$N$', labelpad=2)
plt.ylabel(r'$d_n$ (relative error)')
plt.xlim(right=1e13)
plt.yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8])
plt.savefig('figure_chain_diagram_convergence.pdf')
#plt.show()

# -- Asymptotic plot

plt.figure(figsize=(3.25, 2.25))

gs = GridSpec(
    1, 1,
    width_ratios=[1],
    height_ratios=[1],
    wspace=0.0, hspace=0.45,
    bottom=0.18, top=0.99,
    left=0.16, right=0.99,
    )

plt.subplot(gs[0])
    
for r in sorted(results, key=lambda r: r.parm.order):
    if r.parm.order < 4: continue
    exact = np.exp(-3/2)
    rel_err = np.abs((r.results - exact) / exact)
    plt.plot(r.N_samples_list, rel_err, '-', label=f'{r.parm.order}', alpha=0.75)

n = np.logspace(0.5, 8.75)
plt.plot(n, 2/n, '--k', lw=1.5, alpha=0.5, label='$\sim N_{qMC}^{-1}$')
    
plt.loglog([], [])
plt.legend(title='Diagram order', ncol=1, loc='upper right')
plt.grid(True)
plt.axis('equal')
plt.xlabel(r'$N_{qMC}$')
plt.ylabel(r'$d_n - d_\infty$')
plt.xlim(right=1e13)
plt.savefig('figure_chain_diagram_convergence_asymptotic.pdf')
#plt.show()
