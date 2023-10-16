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

using MPI; MPI.Init()
using HDF5; h5 = HDF5

using LinearAlgebra: diag, diagm
using MD5

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.utility: ph_conj
using QInchworm.ppgf
using QInchworm.expansion: Expansion, InteractionPair

using QInchworm.inchworm: inchworm!
using QInchworm.mpi: ismaster

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

function run_hubbard_dimer(nτ, orders, orders_bare, N_samples)

    β = 10.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0
    μ_bethe = 0.25

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

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), ph_conj(Δ))
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), ph_conj(Δ))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    ρ_0 = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P0), ed), ed)

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    ppgf.normalize!(expansion.P, β)
    ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P), ed), ed)

    global ρ_nca, ρ_oca, ρ_tca, ρ_exa

    diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
    diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
    diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
    diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

    return diff_exa, diff_nca, diff_oca, diff_tca
end

function run_nτ_calc(nτ, orders, N_sampless)

    orders_bare = orders

    # Do calculation here
    diffs_exa = Array{Float64}(undef, length(N_sampless))
    diffs_nca = Array{Float64}(undef, length(N_sampless))
    diffs_oca = Array{Float64}(undef, length(N_sampless))
    diffs_tca = Array{Float64}(undef, length(N_sampless))

    diff_0_exa, diff_0_nca, diff_0_oca, diff_0_tca =
        run_hubbard_dimer(nτ, orders, orders_bare, 0)

    for (idx, N_samples) in enumerate(N_sampless)
        diffs_exa[idx], diffs_nca[idx], diffs_oca[idx], diffs_tca[idx] =
            run_hubbard_dimer(nτ, orders, orders_bare, N_samples)
    end

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs_exa)))
        max_order = maximum(orders)
        filename = "data_FH_bethe_ntau_$(nτ)_maxorder_$(max_order)_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            g = h5.create_group(fid, "data")

            h5.attributes(g)["ntau"] = nτ
            h5.attributes(g)["diff_0_exa"] = diff_0_exa
            h5.attributes(g)["diff_0_nca"] = diff_0_nca
            h5.attributes(g)["diff_0_oca"] = diff_0_oca
            h5.attributes(g)["diff_0_tca"] = diff_0_tca

            g["orders"] = collect(orders)
            g["orders_bare"] = collect(orders_bare)
            g["N_sampless"] = N_sampless

            g["diffs_exa"] = diffs_exa
            g["diffs_nca"] = diffs_nca
            g["diffs_oca"] = diffs_oca
            g["diffs_tca"] = diffs_tca
        end
    end
end

nτs = 2 .^ (4:12)
N_sampless = 2 .^ (4:15)
orderss = [0:2, 0:3]

if ismaster()
    @show nτs
    @show N_sampless
    @show orderss
end

for nτ in nτs
    for orders in orderss
        run_nτ_calc(nτ, orders, N_sampless)
    end
end
