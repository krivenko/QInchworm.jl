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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Hugo U. R. Strand, Igor Krivenko

import time
import numpy as np

from h5 import HDFArchive

import triqs.operators as op
import triqs.utility.mpi as mpi

from triqs_cthyb import Solver

from pydlr import kernel
from scipy.integrate import quad

def eval_semi_circ_tau(tau, beta, h, t):
    def I(x):
        k = kernel(np.array([tau])/beta, beta*np.array([x]))
        return -2 / np.pi / t**2 * k[0,0]
    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    return g

eval_semi_circ_tau = np.vectorize(eval_semi_circ_tau)


def calc_single_fermion(
        beta=8., e0=0., V=0.25, t_bethe=1., mu_bethe=1.0,
        n_cycles=1e5, seed=1337, delta_tau=None, n_ref=None):

    t1 = time.time()
    
    gf_struct = [("0", 1)]

    S = Solver(
        beta=beta,
        gf_struct=gf_struct,
        n_iw=125,
        n_tau=2001,
        n_l=30,
        delta_interface=True)

    H = e0 * op.n("0", 0)

    tau = np.array([ float(t) for t in S.Delta_tau["0"].mesh])

    if delta_tau is None:
        from pydlr import dlr
        d = dlr(lamb=100., eps=1e-10)

        tau_i = d.get_tau(beta)
        delta_i = V**2 * eval_semi_circ_tau(tau_i, beta, mu_bethe, t_bethe)
        delta_i = delta_i.reshape((len(d), 1, 1))
        delta_x = d.dlr_from_tau(delta_i)
        g_x = d.dyson_dlr(np.array([[e0]]), delta_x, beta)
        n_ref = -d.eval_dlr_tau(g_x, np.array([beta]), beta)[0, 0][0]
        if mpi.rank == 0: print(f'n_ref = {n_ref:16.16f}')

        delta_tau = V**2 * eval_semi_circ_tau(tau, beta, mu_bethe, t_bethe)


    if mpi.rank == 0:
        print(f'tau = {tau}')
        
    for s, delta in S.Delta_tau:
        delta.data[:, 0, 0] = delta_tau

    seed = seed + 13 * mpi.rank
    n_cycles = n_cycles // mpi.size
        
    S.solve(
        h_int=H, h_loc0=0*H,
        n_cycles=int(n_cycles),
        n_warmup_cycles=int(n_cycles)//4,
        measure_pert_order=True,
        use_norm_as_weight=True,
        measure_density_matrix=True,
        random_seed=seed,
        )

    if mpi.rank == 0:
        print(S.perturbation_order_total.data[:15])
        print(S.density_matrix)
        diff = np.abs(0.5 - S.density_matrix[0][0, 0])
        print(f'N {S.solve_parameters["n_cycles"]:1.1E} err {diff:2.2E}')

    t2 = time.time()
    S.time = t2 - t1
    return S, delta_tau, n_ref


if __name__ == "__main__":

    seeds = 34788 + 928374 * np.arange(0, 32)

    delta_tau, n_ref = None, None

    ps = []
    times = []

    n_cycless = 2**np.arange(4, 20)
    #n_cycless = 2**np.arange(4, 25)

    if mpi.rank == 0: print(f'n_cycless = {n_cycless}')
    
    for n_cycles in n_cycless:
        for seed in seeds:
            p, delta_tau, n_ref = calc_single_fermion(
                n_cycles=n_cycles, seed=seed,
                delta_tau=delta_tau, n_ref=n_ref)
            times.append(p.time)
            ps.append(p)

    filename = 'data_bethe_cthyb_new.h5'
    if mpi.is_master_node():
        print(f'n_ref = {n_ref:16.16E}')
        with HDFArchive(filename, 'w') as a:
            a['ps'] = ps
            a['times'] = times
            a['n_ref'] = n_ref
            a['mpi_size'] = mpi.size
