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
# Authors: Igor Krivenko, Hugo U. R. Strand

using MPI; MPI.Init()
using HDF5; h5 = HDF5

using LinearAlgebra: diag, diagm

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.utility: ph_conj
using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.inchworm: inchworm!
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.mpi: ismaster

# Solve non-interacting two fermion AIM coupled to
# semi-circular (Bethe lattice) hybridization functions.
#
# Compare to numerically exact results for the 1st, 2nd and 3rd
# order dressed self-consistent expansion for the many-body
# density matrix (computed using DLR elsewhere).
#
# Note that the 1, 2, 3 order density matrix differs from the
# exact density matrix of the non-interacting system, since
# the low order expansions introduce "artificial" effective
# interactions between hybridization insertions.

function ρ_from_n_ref(n_ref)
    return diagm([(1 - n_ref) * (1 - n_ref),
                  n_ref * (1 - n_ref),
                  n_ref * (1 - n_ref),
                  n_ref * n_ref])
end

# Reference data from DLR
const ρ_exa = ρ_from_n_ref(0.5460872495307262)
const ρ_nca = diagm([0.1961713995875524,
                     0.2474226001525296,
                     0.2474226001525296,
                     0.3089834001073883])
const ρ_oca = diagm([0.2018070389569783,
                     0.2476929924482211,
                     0.2476929924482211,
                     0.3028069761465793])
const ρ_tca = diagm([0.205163794520457,
                     0.2478638876741985,
                     0.2478638876741985,
                     0.2991084301311462])

function run_hubbard_dimer(nτ, orders, orders_bare, N_samples, μ_bethe, interpolation)

    β = 10.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0

    # ED solution

    H_imp = -μ * (op.n(1) + op.n(2))

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.bethe_dos(ϵ=μ_bethe, t=0.5), grid)

    # Pseudo Particle Strong Coupling Expansion

    if interpolation
        ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
        ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(ph_conj(Δ)))
        ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), SplineInterpolatedGF(Δ))
        ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), SplineInterpolatedGF(ph_conj(Δ)))
        expansion = Expansion(ed,
                              grid,
                              [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd],
                              interpolate_ppgf=true)
    else
        ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), ph_conj(Δ))
        ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
        ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), ph_conj(Δ))
        expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])
    end

    ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    if interpolation
        # Extract a plain (non-interpolated) version of expansion.P
        P = [deepcopy(P_int.GF) for P_int in expansion.P]

        normalize!(P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(P), ed), ed)
    else
        normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(expansion.P), ed), ed)
    end

    global ρ_nca, ρ_oca, ρ_tca, ρ_exa

    diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
    diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
    diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
    diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

    ρ_ref = real.(diag.([ρ_0, ρ_nca, ρ_oca, ρ_tca, ρ_exa]))
    diffs = [diff_nca, diff_oca, diff_tca, diff_exa]

    return real(diag(ρ_wrm)), ρ_ref, diffs
end

μ_bethe = 0.25
N_samples = 2^16
nτ_list = 2 .^ (3:10)

if ismaster()
    h5.h5open("data.h5", "w") do fid
        g = h5.create_group(fid, "data")
        h5.attributes(g)["mu_bethe"] = μ_bethe
        h5.attributes(g)["N_samples"] = N_samples
    end
end

for orders in [0:1, 0:2, 0:3]

    diffs_non_interp = zeros(Float64, length(nτ_list), 4)
    diffs_interp = zeros(Float64, length(nτ_list), 4)

    for (i, nτ) in pairs(nτ_list)
        ρ_wrm, ρ_ref, diffs =
            run_hubbard_dimer(nτ, orders, orders, N_samples, μ_bethe, false)
        diffs_non_interp[i, :] = diffs
        ρ_wrm, ρ_ref, diffs =
            run_hubbard_dimer(nτ, orders, orders, N_samples, μ_bethe, true)
        diffs_interp[i, :] = diffs
    end

    if ismaster()
        println("orders = ", orders)
        print("diffs_non_interp = ")
        display(diffs_non_interp)
        print("diffs_interp = ")
        display(diffs_interp)

        h5.h5open("data.h5", "cw") do fid
            g = h5.create_group(fid["data"], "orders_$orders")
            h5.attributes(g)["ntau_list"] = nτ_list
            g["orders"] = collect(orders)
            g["diffs_non_interp"] = diffs_non_interp
            g["diffs_interp"] = diffs_interp
        end
    end
end
