import numpy as np

from h5 import HDFArchive

import triqs.operators as op
import triqs.utility.mpi as mpi

from matplotlib.gridspec import GridSpec

from triqs_cthyb import Solver

from triqs_tprf.ParameterCollection import ParameterCollection
from triqs_tprf.ParameterCollection import ParameterCollections

if __name__ == "__main__":

    #filename = "data_bethe_ntau_64_maxorder_3_md5_bbddca17a291cfb61f28a3e4e284013b.h5"
    #filename = "data_bethe_ntau_64_maxorder_5_md5_ca76db0c5ac5a406b472cc0a5313e5ce.h5"
    filename = "data_bethe_ntau_128_maxorder_5_md5_29e967aaf42881c103bce5b53fca1fa1.h5"
    print(f'--> Loading: {filename}')
    with HDFArchive(filename, 'r') as a:
        d = a['data']
        pto_hist_wrm = d['pto_hists'][-1]

    print(f'pto_hist_wrm = {pto_hist_wrm}')
    #exit()
    
    #filename = 'data_bethe_ntau_1024_maxorder_1_md5_68e17e11280c9c0cb659042ff7a8ef7d.h5'
    #filename = 'data_bethe_ntau_2048_maxorder_1_md5_1bcc2a4d808c1b5e5e924af06a0afb72.h5'
    filename = 'data_bethe_ntau_4096_maxorder_1_md5_d3e0ae1f4957e91a05861c96af3f3162.h5'

    print(f'--> Loading: {filename}')
    with HDFArchive(filename, 'r') as a:
        d = a['data']
        diffs = d['diffs']
        N_sampless = d['N_sampless']

    filename = 'data_bethe_cthyb.h5'
    print(f'--> Loading: {filename}')
    with HDFArchive(filename, 'r') as a:
        ps = a['ps']
        n_ref = a['n_ref']

    print(f'n_ref 1 = {n_ref:16.16f}')
    n_ref = 1. - n_ref
    print(f'n_ref 2 = {n_ref:16.16f}')
    #print(ps)
    
    ds = []
    for p in ps:
        d = ParameterCollection()
        #print(p.solve_parameters)
        #print(p.solve_parameters['n_cycles'])
        #print(p.solve_parameters['length_cycle'])
        d.n_samples = p.solve_parameters['n_cycles'] * p.solve_parameters['length_cycle']
        #d.n_samples = p.solve_parameters['n_cycles']
        d.density_matrix = p.density_matrix
        pto_max = 7
        d.pto = p.perturbation_order_total.data[:pto_max]
        ds.append(d)
        
    ds = ParameterCollections(ds)
    #print(ds.density_matrix.shape)
    #print(ds.n_samples)

    n_samples = np.unique(ds.n_samples)
    print(n_samples)
    #exit()

    err_avgs = np.zeros((len(n_samples)))
    for idx, ns in enumerate(n_samples):
        #print(f'ns = {ns}, idx = {idx}')
        meas_idx = ds.n_samples == ns
        #print(f'meas_idx = {meas_idx}')
        meas = ds.density_matrix[meas_idx]
        #print(meas.shape)
        n = meas.shape[0]
        err_avgs[idx] = np.sum(np.squeeze(np.abs(meas[:, 0] - n_ref)), axis=0) / n

    errs = np.abs(np.squeeze(ds.density_matrix[:, 0]) - n_ref)

    def target_function(C):
        x = n_samples
        return np.sum(np.abs(err_avgs/(C/np.sqrt(x)) - 1.))

    from scipy.optimize import minimize
    res = minimize(target_function, 0.1)
    print(res)

    def target_function_qmc(C):
        x = N_sampless
        y = diffs

        n_cut = 5
        x, y = x[:n_cut], y[:n_cut]
        
        return np.sum(np.abs(y/(C/x) - 1.))

    from scipy.optimize import minimize
    res = minimize(target_function, 0.1)
    print(res)

    res_qmc = minimize(target_function_qmc, 0.1)
    print(res_qmc)
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(3.25, 4))

    gs = GridSpec(
        2, 1,
        width_ratios=[1],
        height_ratios=[1, 0.5],
        wspace=0.0, hspace=0.35,
        bottom=0.10, top=0.97,
        left=0.25, right=0.98,
        )
    
    #subp = [2, 1, 1]
    #plt.subplot(*subp); subp[-1] += 1
    plt.subplot(gs[0, 0])

    delta = np.logspace(-5, 0, num=10)
    C = res_qmc.x
    plt.plot(C/delta, delta, ':k', lw=0.5,
             #label=r'$1/N$'
             )

    #x = np.logspace(0, 8, num=10)
    #plt.plot(x, res.x/np.sqrt(x), '--k', lw=0.5, label=r'$\sim 1/\sqrt{N}$')

    delta = np.logspace(-4.5, 0, num=10)
    plt.plot(res.x**2 / delta**2, delta, '--k', lw=0.5,
             #label=r'$1/\sqrt{N}$',
             )
    
    plt.plot(N_sampless, diffs, '.-', alpha=0.75, label='Quasi Monte Carlo')

    l = plt.plot(ds.n_samples, errs, '.', alpha=0.02)
    color = l[0].get_color()
    plt.plot(n_samples, err_avgs, '-', color=color, alpha=0.75)
    plt.plot([], [], '.-', color=color, alpha=0.75, label='Monte Carlo')

    plt.plot(2e9, 1e-3, 'o', color='white')
    plt.plot(2e9, 1e2, 'o', color='white')
    plt.text(1e7, 2e-5, r'$\sim 1/\sqrt{N}$', ha='left', va='top')
    plt.text(2e5, 7e-6, r'$\sim 1/N$', ha='center', va='top')
    
    plt.loglog([],[])
    #plt.semilogx([], [])
    #plt.grid(True)
    plt.xlabel(r'Samples $N$')
    plt.ylabel(r'Error in  $\rho$')
    plt.legend(loc='upper center')
    #plt.ylim([1e-4, 1e1])
    #plt.xlim([1e-2, 1e9])
    plt.axis('equal')
    #plt.ylim([1e-5, 1e0])

    #plt.subplot(*subp); subp[-1] += 1
    plt.subplot(gs[1, 0])

    p = ps[-1]
    #print(dir(p))
    #print(p.perturbation_order_total)


    n = 0
    pto = np.zeros_like(ds.objects[-1].pto)
    for d in ds:
        if d.n_samples == n_samples[-1]:
            pto += d.pto
            n += 1
    pto /= n
        
    #pto = p.perturbation_order_total
    #pto = pto.data[:20]
    
    #print(pto.shape)
    #print(pto)
    #print(dir(pto))
    #import matplotlib.pyplot as plt
    x = np.arange(0, len(pto))
    pto /= np.sum(pto)

    color_wrm = plt.plot([], [])[0].get_color()
    color_cth = plt.plot([], [])[0].get_color()

    plt.bar(x, pto, alpha=0.75, color=color_cth)
    pto_hist_wrm = np.abs(pto_hist_wrm)
    pto_hist_wrm /= np.sum(pto_hist_wrm)
    plt.bar(range(len(pto_hist_wrm)), pto_hist_wrm, alpha=0.75, color=color_wrm)

    plt.semilogy([], [])
    #plt.ylim(top=1)
    plt.xlabel('Perturbation order')
    plt.ylabel('Probabilbity')
    plt.xticks(np.arange(0, pto_max), labels=[f'{i:d}' for i in range(0, pto_max)])
    plt.ylim(bottom=5e-8)
    
    plt.tight_layout()
    plt.savefig('figure_mc_convergence.pdf')
    plt.show()
