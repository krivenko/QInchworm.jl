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

using MD5
using HDF5; h5 = HDF5

using LinearAlgebra: diag
using Printf

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.utility: ph_conj
using QInchworm.ppgf
using QInchworm.expansion: Expansion, InteractionPair

using QInchworm.inchworm: inchworm!
using QInchworm.mpi: ismaster

function run_hubbard_dimer(nτ, orders, orders_bare, N_samples)

    β = 1.0
    U = 4.0
    ϵ_1, ϵ_2 = 0.0 - 0.5*U, 2.0
    V_1 = 0.5
    V_2 = 0.5

    # ED solution

    H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))

    H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) +
        V_1 * (op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1)) +
        V_2 * (op.c_dag(2) * op.c(4) + op.c_dag(4) * op.c(2))

    soi_dimer = ked.Hilbert.SetOfIndices([[1], [2], [3], [4]])
    ed_dimer = ked.EDCore(H_dimer, soi_dimer)

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    ρ_ref = Array{ComplexF64}(reduced_density_matrix(ed_dimer, soi, β))

    # Hybridization propagator

    Δ_1 = V_1^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ_2), grid)
    Δ_2 = V_2^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ_2), grid)

    # Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ_1)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), ph_conj(Δ_1))
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ_2)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), ph_conj(Δ_2))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    ρ_0 = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P0), ed), ed)

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    ppgf.normalize!(expansion.P, β)
    ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P), ed), ed)
    diff = maximum(abs.(ρ_ref - ρ_wrm))

    if ismaster()
        @printf "ρ_0   = %16.16f %16.16f %16.16f %16.16f \n" real(diag(ρ_0))...
        @printf "ρ_ref = %16.16f %16.16f %16.16f %16.16f \n" real(diag(ρ_ref))...
        @printf "ρ_wrm = %16.16f %16.16f %16.16f %16.16f \n" real(diag(ρ_wrm))...
        @show diff
    end

    return diff
end

function run_nτ_calc(nτ, orders, N_sampless)

    orders_bare = orders

    # Do calculation here

    diff_0 = run_hubbard_dimer(nτ, orders, orders_bare, 0)
    diffs = [run_hubbard_dimer(nτ, orders, orders_bare, N_samples)
             for N_samples in N_sampless]

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        max_order = maximum(orders)
        filename = "data_FH_dimer_ntau_$(nτ)_maxorder_$(max_order)_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            g = h5.create_group(fid, "data")

            h5.attributes(g)["ntau"] = nτ
            h5.attributes(g)["diff_0"] = diff_0

            g["orders"] = collect(orders)
            g["orders_bare"] = collect(orders_bare)
            g["N_sampless"] = N_sampless

            g["diffs"] = diffs
        end
    end
end

nτs = 2 .^ (4:8)
orderss = [0:2, 0:3]
N_sampless = 2 .^ (15:17)

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
