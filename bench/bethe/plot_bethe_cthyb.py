# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
#
# QInchworm.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# QInchworm.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/.
#
# Authors: Hugo U. R. Strand, Igor Krivenko

import argparse

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from h5 import HDFArchive
from triqs_cthyb import Solver

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('qinchworm_filename',
                        type=str,
                        help="Name of the QInchworm HDF5 results file")
    parser.add_argument('--cthyb_filename',
                        type=str,
                        default="data_bethe_cthyb.h5",
                        help="Name of the CTHYB HDF5 results file")

    args = parser.parse_args()

    print(f'--> Loading: {args.qinchworm_filename}')
    with HDFArchive(args.qinchworm_filename, 'r') as a:
        d = a['data']
        diffs = d['diffs']
        N_sampless = d['N_sampless']
        pto_hist_wrm = d['pto_hists'][-1]

    print(f'--> Loading: {args.cthyb_filename}')
    with HDFArchive(args.cthyb_filename, 'r') as a:
        ps = a['ps']
        n_ref = a['n_ref']

    print(f'n_ref 1 = {n_ref:16.16f}')
    n_ref = 1. - n_ref
    print(f'n_ref 2 = {n_ref:16.16f}')

    ds = []
    pto_max = 7
    for p in ps:
        d = {}
        d["n_samples"] = p.solve_parameters['n_cycles'] * \
                         p.solve_parameters['length_cycle']
        d["density_matrix"] = p.density_matrix
        d["pto"] = p.perturbation_order_total.data[:pto_max]
        ds.append(d)

    def slice_ds(name):
        return np.array([d[name] for d in ds])

    n_samples = np.unique(slice_ds("n_samples"))
    print(n_samples)

    err_avgs = np.zeros((len(n_samples)))
    for idx, ns in enumerate(n_samples):
        meas_idx = slice_ds("n_samples") == ns
        meas = slice_ds("density_matrix")[meas_idx]
        n = meas.shape[0]
        err_avgs[idx] = np.sum(np.squeeze(np.abs(meas[:, 0] - n_ref)), axis=0) / n

    errs = np.abs(np.squeeze(slice_ds("density_matrix")[:, 0]) - n_ref)

    def target_function(C):
        x = n_samples
        return np.sum(np.abs(err_avgs / (C / np.sqrt(x)) - 1.))

    res = minimize(target_function, 0.1)
    print(res)

    def target_function_qmc(C):
        x = N_sampless
        y = diffs

        n_cut = 5
        x, y = x[:n_cut], y[:n_cut]

        return np.sum(np.abs(y / (C / x) - 1.))

    res = minimize(target_function, 0.1)
    print(res)

    res_qmc = minimize(target_function_qmc, 0.01)
    print(res_qmc)

    plt.figure(figsize=(3.25, 2.5))

    gs = GridSpec(
        2, 1,
        width_ratios=[1],
        height_ratios=[1, 0.3],
        wspace=0.0, hspace=0.45,
        bottom=0.14, top=0.99,
        left=0.18, right=0.98,
        )

    plt.subplot(gs[0, 0])

    N = np.logspace(0.2, 6.5, num=10)
    plt.plot(N, res_qmc.x / N, ':k', lw=0.5)

    N = np.logspace(0.2, 14, num=10)
    plt.plot(N, res.x / np.sqrt(N), '--k', lw=0.5,)

    plt.plot(N_sampless, diffs, '.-', alpha=0.75, label='Quasi Monte Carlo')

    l = plt.plot(slice_ds("n_samples"), errs, '.', alpha=0.02)
    color = l[0].get_color()
    plt.plot(n_samples, err_avgs, '-', color=color, alpha=0.75)
    plt.plot([], [], '.-', color=color, alpha=0.75, label='Monte Carlo')

    plt.text(1e8, 2e-4, r'$\propto 1/\sqrt{N}$', ha='left', va='top')
    plt.text(2e7, 5e-8, r'$\propto 1/N$', ha='center', va='top')

    plt.loglog([],[])
    plt.xlabel(r'Samples $N$')
    plt.ylabel(r'Error in $\rho$')
    plt.legend(loc='best')

    plt.xlim(left=1, right=1e15)
    plt.ylim(bottom=1e-8)

    plt.subplot(gs[1, 0])

    p = ps[-1]

    n = 0
    pto = np.zeros_like(ds[-1]["pto"])
    for d in ds:
        if d["n_samples"] == n_samples[-1]:
            pto += d["pto"]
            n += 1
    pto /= n

    x = np.arange(0, len(pto))
    pto /= np.sum(pto)

    color_wrm = plt.plot([], [])[0].get_color()
    color_cth = plt.plot([], [])[0].get_color()

    plt.bar(x, pto, alpha=0.75, color=color_cth)
    pto_hist_wrm = np.abs(pto_hist_wrm)
    pto_hist_wrm /= np.sum(pto_hist_wrm)
    plt.bar(range(len(pto_hist_wrm)), pto_hist_wrm, alpha=0.75, color=color_wrm)

    plt.semilogy([], [])
    plt.xlabel('Perturbation order')
    plt.ylabel('Probabilbity')
    plt.xticks(np.arange(0, pto_max), labels=[f'{i:d}' for i in range(0, pto_max)])
    plt.ylim(bottom=5e-8, top=10)
    plt.yticks([1e-6, 1e-3, 1e0])

    plt.tight_layout()
    plt.savefig('figure_mc_convergence.pdf')
    plt.show()
