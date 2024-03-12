# QInchworm.jl
#
# Copyright (C) 2021-2024 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Hugo U. R. Strand, Igor Krivenko

import glob
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
    parser.add_argument('--qinchworm_filenames',
                        nargs='+',
                        default=['data_bethe_ntau_*'],
                        help="Name of the QInchworm HDF5 results files")
    parser.add_argument('--cthyb_filename',
                        type=str,
                        #default="data_bethe_cthyb.h5",
                        default="data_bethe_cthyb_new.h5",
                        help="Name of the CTHYB HDF5 results file")

    args = parser.parse_args()

    args.qinchworm_filenames = sum(list(map(
        glob.glob, args.qinchworm_filenames)), start=[])

    diffs, times, N_sampless, N_seqss = [], [], [], []
    for filename in args.qinchworm_filenames:
        print(f'--> Loading: {filename}')
        with HDFArchive(filename, 'r') as a:
            d = a['data']
            diffs.append(list(d['diffs']))
            times.append(list(d['times']))
            if 'N_seqs' in d.keys():
                N_seqs = d['N_seqs']
            else:
                N_seqs = 1
            N_seqss.append(N_seqs)

            N_sampless.append( list(np.array(N_seqs * d['N_sampless'])) )
            #N_sampless.append( list(np.array(d['N_sampless'])) )

    #diffs, N_sampless = np.array(diffs), np.array(N_sampless)
    #sidx = np.argsort(N_sampless)
    #diffs, N_sampless = diffs[sidx], N_sampless[sidx]

    print(f'N_sampless = {N_sampless}')
    print(f'diffs = {diffs}')
    print(f'times = {times}')

    print(f'--> Loading: {args.cthyb_filename}')
    with HDFArchive(args.cthyb_filename, 'r') as a:
        ps = a['ps']
        n_ref = 1 - a['n_ref']
        times_cthyb = a['times']

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

    def target_function_qmc(C):
        x = N_sampless[0]
        y = diffs[0]

        n_cut = 5
        x, y = x[:n_cut], y[:n_cut]

        return np.sum(np.abs(y / (C / x) - 1.))

    res = minimize(target_function, 0.1)
    res_qmc = minimize(target_function_qmc, 0.001)

    # -- Timing vs N

    plt.figure(figsize=(3.25, 2.))
    plt.plot(slice_ds("n_samples"), times_cthyb, 'o', alpha=0.1, label='CTHYB')
    for N_samples, time, N_seqs in zip(N_sampless, times, N_seqss):
        print(time)
        print(N_sampless)
        plt.plot(N_sampless[0], time, '-s', label='Inchworm', alpha=0.75)

    plt.legend(loc='best')
    plt.loglog([], [])
    plt.grid(True)
    plt.xlabel(r'Samples $N$')
    plt.ylabel(r'Time (sec)')

    #plt.show(); exit()

    # -- Convergence vs N

    for label in ['1', '2']:
        plt.figure(figsize=(3.25, 2.))

        gs = GridSpec(
            1, 1,
            width_ratios=[1],
            height_ratios=[1],
            wspace=0.0, hspace=0.45,
            bottom=0.14, top=0.99,
            left=0.16, right=0.99,
            )

        N = np.logspace(0.2, 14, num=10)
        plt.plot(N, res.x / np.sqrt(N), '--k', lw=0.5)

        if label == '2':
            for N_samples, diff, N_seqs in zip(N_sampless, diffs, N_seqss):
                plt.plot(N_samples, diff, 'o-', markersize=2.5, alpha=0.75, label=f'Quasi Monte Carlo Nseq = {N_seqs}')
            plt.text(2e7, 5e-8, r'$\propto 1/N$', ha='center', va='top')
            N = np.logspace(0.2, 6.5, num=10)
            plt.plot(N, res_qmc.x / N, ':k', lw=0.5)
        else:
            plt.plot([], [], 'o-', markersize=2.5, alpha=0.75, label='Quasi Monte Carlo')

        l = plt.plot(slice_ds("n_samples"), errs, 's', alpha=0.02, markersize=2.5)
        color = l[0].get_color()
        plt.plot(n_samples, err_avgs, '-', color=color, alpha=0.75)
        plt.plot([], [], 's-', markersize=2.5, color=color, alpha=0.75, label='Monte Carlo')

        plt.text(1e8, 2e-4, r'$\propto 1/\sqrt{N}$', ha='left', va='top')

        plt.loglog([],[])
        plt.grid(True)
        plt.xlabel(r'Samples $N$')
        plt.ylabel(r'$|\rho - \rho_{exact}|$')
        plt.legend(loc='upper right')

        plt.axis('equal')
        plt.ylim([1e-8, 1e0])

        plt.tight_layout()
        plt.savefig(f'figure_mc_convergence_{label}.pdf')

    plt.show()
