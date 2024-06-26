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


class Dummy():
    def __init__(self):
        pass


def read_cthyb_data(filename, pto_max=7):
    
    print(f'--> Loading: {filename}')
    
    with HDFArchive(filename, 'r') as a:
        ps = a['ps']
        n_ref = 1 - a['n_ref']
        cthyb_mpi_size = int(a['mpi_size'])
        times_cthyb = np.array(a['times']) * cthyb_mpi_size
        #times_cthyb = np.array(a['times'])

    ds = []

    for p in ps:
        d = {}
        d["n_samples"] = p.solve_parameters['n_cycles'] * \
                         cthyb_mpi_size        
                         #p.solve_parameters['length_cycle'] * cthyb_mpi_size        
        d["density_matrix"] = p.density_matrix
        d["pto"] = p.perturbation_order_total.data[:pto_max]
        ds.append(d)

    def slice_ds(name):
        return np.array([d[name] for d in ds])

    n_samples = np.unique(slice_ds("n_samples"))

    print(f'cthyb mpi_size = {cthyb_mpi_size}')
    print(f'cthyb n_samples = {n_samples}')
    #print(f'cthyb times = {times_cthyb}')

    err_avgs = np.zeros((len(n_samples)))
    times_cthyb_avgs = np.zeros((len(n_samples)))
    
    for idx, ns in enumerate(n_samples):
        meas_idx = slice_ds("n_samples") == ns
        meas = slice_ds("density_matrix")[meas_idx]
        n = meas.shape[0]
        err_avgs[idx] = np.sum(np.squeeze(np.abs(meas[:, 0] - n_ref)), axis=0) / n

        meas = np.array(times_cthyb)[meas_idx]
        times_cthyb_avgs[idx] = np.mean(meas)

    errs = np.abs(np.squeeze(slice_ds("density_matrix")[:, 0]) - n_ref)
    
    d = Dummy()

    d.filename = filename
    d.ds = ds

    d.errs = errs
    d.times = times_cthyb
    d.n_samples = n_samples
    d.n_samples_slice = slice_ds("n_samples")

    d.err_avgs = err_avgs
    d.times_avgs = times_cthyb_avgs
    
    return d


def merge_cthyb_data(cthybs):

    cthyb = cthybs[0]

    for c in cthybs[1:]:
        for key, val in cthyb.__dict__.items():
            #print(f'{key} {type(val)}')
            if type(val) == np.ndarray:
                cthyb.__dict__[key] = np.concatenate((val, c.__dict__[key]))
            if type(val) == list:
                cthyb.__dict__[key] = val + c.__dict__[key]

    return cthyb


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--qinchworm_filenames',
                        nargs='+',
                        default=['data_bethe_ntau_*'],
                        help="Name of the QInchworm HDF5 results files")
    parser.add_argument('--cthyb_filenames',
                        nargs='+',
                        #type=str,
                        #default="data_bethe_cthyb.h5",
                        #default="data_bethe_cthyb_new.h5",
                        #default="data_bethe_cthyb_new_7_25.h5",
                        #default="data_bethe_cthyb_new_7_30.h5",
                        default=[
                            "data_bethe_cthyb_new_2_7.h5",
                            "data_bethe_cthyb_new_7_25.h5",
                            "data_bethe_cthyb_new_11_25.h5",
                            "data_bethe_cthyb_new_25_35.h5",
                        ],
                        help="Name of the CTHYB HDF5 results file")

    args = parser.parse_args()

    args.qinchworm_filenames = sum(list(map(
        glob.glob, args.qinchworm_filenames)), start=[])
    
    diffs, times, N_sampless, N_seqss, orders = [], [], [], [], []
    for filename in args.qinchworm_filenames:
        print(f'--> Loading: {filename}')
        with HDFArchive(filename, 'r') as a:
            d = a['data']

            mpi_ranks = int(d['mpi_ranks'])
            N_seqs = d['N_seqs'] if 'N_seqs' in d.keys() else 1

            diff = list(d['diffs'])
            time = mpi_ranks * np.array(d['times'])
            order = np.max(list(d['orders']))
            N_samples = list(np.array(N_seqs * d['N_sampless']))
            #N_samples = list(d['N_sampless'])
            
            diffs.append(diff)
            times.append(time)
            N_seqss.append(N_seqs)
            orders.append(order)
            N_sampless.append(N_samples)

    #diffs, N_sampless = np.array(diffs), np.array(N_sampless)
    #sidx = np.argsort(N_sampless)
    #diffs, N_sampless = diffs[sidx], N_sampless[sidx]

    print(f'N_sampless = {N_sampless}')
    print(f'diffs = {diffs}')
    print(f'times = {times}')
    print(f'orders = {orders}')
    #exit()

    def cut_off_data(cthyb, n_min=2**9 // 4, n_cut=2**13 // 4):

        print(f'--> Cutting: {cthyb.filename}')
        
        if n_min in cthyb.n_samples:

            sidx = cthyb.n_samples < n_cut
            for key in ['n_samples', 'times_avgs', 'err_avgs']:
                val = getattr(cthyb, key)
                setattr(cthyb, key, val[sidx])
            
            sidx = cthyb.n_samples_slice < n_cut
            for key in ['n_samples_slice', 'times', 'errs']:
                val = getattr(cthyb, key)
                setattr(cthyb, key, val[sidx])
            
        return cthyb

    cthybs = [ cut_off_data(read_cthyb_data(filename)) for filename in args.cthyb_filenames ]
    
    cthyb = merge_cthyb_data(cthybs)
        
    def target_function(C):
        x = cthyb.n_samples
        return np.sum(np.abs(cthyb.err_avgs / (C / np.sqrt(x)) - 1.))

    def target_function_qmc(C):
        x = N_sampless[0]
        y = diffs[0]

        n_cut = 5
        x, y = x[:n_cut], y[:n_cut]

        return np.sum(np.abs(y / (C / x) - 1.))

    res = minimize(target_function, 0.1)
    res_qmc = minimize(target_function_qmc, 0.001)

    # -- Timing vs accuracy

    plt.figure(figsize=(3.25, 2.))

    #time_scale = 1 / 60 / 60 / 24 
    time_scale = 1 / 60 / 60 
    
    for N_samples, diff, time, N_seqs, order in zip(N_sampless, diffs, times, N_seqss, orders):
        #plt.plot(time, diff, '-s', label=f'Inchworm O{order}', alpha=0.75)
        plt.plot(time * time_scale, diff, 'o-', markersize=2.5, alpha=0.75,
                 label=f'Quasi Monte Carlo',
                 )

    l = plt.plot(cthyb.times * time_scale, cthyb.errs, 's', alpha=0.02, markersize=2.5)
    color = l[0].get_color()
    plt.plot(cthyb.times_avgs * time_scale, cthyb.err_avgs, '-', color=color, alpha=0.75)
    plt.plot([], [], 's-', markersize=2.5, color=color, alpha=0.75, label='Monte Carlo')

    # linear fit to CTHYB

    n = 10
    x, y = np.log10(cthyb.times_avgs[-n:]), np.log10(cthyb.err_avgs[-n:])
    p = np.polyfit(x, y, 1)
    xf = np.linspace(3, 11, num=10)
    yf = np.polyval(p, xf)

    #plt.plot(10**x, 10**y, 'or')
    plt.plot(10**xf * time_scale, 10**yf, '--k', lw=0.5, zorder=-10)
    
    #plt.plot(cthyb.times, cthyb.errs, 'o', alpha=0.1, label='CTHYB')
    #plt.plot(cthyb.times_avgs, cthyb.err_avgs, '-', alpha=0.75)
    

    plt.legend(loc='best')
    plt.loglog([], [])
    plt.grid(True)
    #plt.ylabel(r'Error')
    plt.ylabel(r'$|\rho - \rho_{exact}|$')
    #plt.xlabel(r'Run-time (core-sec)')
    #plt.xlabel(r'Run-time (core-years)')
    #plt.xlabel(r'Run-time (core-days)')
    plt.xlabel(r'Run-time (core-hours)')

    plt.xlim([1e1 * time_scale, 1e12 * time_scale])
    plt.ylim([1e-9, 1e-3])
    
    plt.tight_layout()
    plt.savefig('figure_mc_convergence_timing_vs_accuracy.pdf')
        
    #plt.show(); exit()
    
    # -- Timing vs N

    plt.figure(figsize=(3.25, 2.))

    plt.plot(cthyb.n_samples_slice, cthyb.times, 'o', alpha=0.75, label='CTHYB')
    
    for N_samples, time, N_seqs, order in zip(N_sampless, times, N_seqss, orders):
        plt.plot(N_samples[1:], time[1:], '-s', label=f'Inchworm O{order}', alpha=0.75)
        
    plt.legend(loc='best')
    plt.loglog([], [])
    plt.grid(True)
    #plt.xlabel(r'Samples $N$')
    plt.xlabel(r'$N$')
    plt.ylabel(r'Time (sec)')

    #plt.show(); exit()

    # -- Convergence vs N
    
    for label in ['1', '2']:
        plt.figure(figsize=(3.25, 2.))

        if True:
            gs = GridSpec(
                1, 1,
                width_ratios=[1],
                height_ratios=[1],
                wspace=0.0, hspace=0.45,
                bottom=0.17, top=0.98,
                left=0.17, right=0.99,
                )
            plt.subplot(gs[0,0])
            
        #N = np.logspace(0.2, 14, num=10)
        N = np.logspace(0.2, 14, num=10)
        plt.plot(N, res.x / np.sqrt(N), '--k', lw=0.5)

        if label == '1':
            alpha = 0.0
            full = 0.0
            qmc_label = None
        else:
            alpha = 0.75
            full = 1.0
            qmc_label = f'Quasi Monte Carlo'
            
        #if label == '2':
        if True:
            for N_samples, diff, N_seqs in zip(N_sampless, diffs, N_seqss):
                l = plt.plot(N_samples, diff, 'o-', markersize=2.5, #alpha=0.75,
                         #label=f'Quasi Monte Carlo Nseq = {N_seqs}',
                         label=qmc_label,
                         alpha=alpha,
                         )
            #plt.text(2e7, 5e-8, r'$\propto 1/N$', ha='center', va='top')
            plt.text(1e7, 5e-8, r'$\propto 1/N$', ha='center', va='top', alpha=full)
            N = np.logspace(0.2, 6.5, num=10)
            plt.plot(N, res_qmc.x / N, ':k', lw=0.5, alpha=full)
        else:
            plt.plot([], [], 'o-', markersize=2.5, alpha=0.75, label='Quasi Monte Carlo')

        if label == '1':
            plt.plot([], [], 'o-', markersize=2.5, alpha=0.75,
                     label='Quasi Monte Carlo', color=l[0].get_color())
            
        l = plt.plot(cthyb.n_samples_slice, cthyb.errs, 's', alpha=0.02, markersize=2.5)
        color = l[0].get_color()
        plt.plot(cthyb.n_samples, cthyb.err_avgs, '-', color=color, alpha=0.75)
        plt.plot([], [], 's-', markersize=2.5, color=color, alpha=0.75, label='Monte Carlo')

        #plt.text(1e8, 2e-4, r'$\propto 1/\sqrt{N}$', ha='left', va='top')
        plt.text(1e12, 2e-6, r'$\propto 1/\sqrt{N}$', ha='left', va='top')

        plt.loglog([],[])
        plt.grid(True)
        #plt.xlabel(r'Samples $N$')
        plt.xlabel(r'$N$')
        plt.ylabel(r'$|\rho - \rho_{exact}|$')
        plt.legend(loc='upper right')


        #plt.yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
        #plt.xticks([1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14])

        plt.yticks([1e0, 1e-4, 1e-8])
        plt.xticks([1e0, 1e4, 1e8, 1e12])
        
        #plt.ylim([1e-8, 1e0])

        #plt.ylim([1e-9, 1e0])

        plt.axis('square')
        plt.ylim(top=1e0)
        plt.xlim(left=1e-1)
        
        plt.tight_layout()
        plt.savefig(f'figure_mc_convergence_{label}.pdf')

    #plt.show()
