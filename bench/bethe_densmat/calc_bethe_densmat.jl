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
using Printf
using MD5

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators

using QInchworm.utility: ph_conj
using QInchworm.ppgf
using QInchworm.expansion: Expansion, InteractionPair

using QInchworm.inchworm: inchworm!
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.mpi: ismaster

using QInchworm.keldysh_dlr: DLRImaginaryTimeGrid, DLRImaginaryTimeGF, ph_conj
import Lehmann; le = Lehmann

function run_bethe(nτ, orders, orders_bare, N_samples; interpolate_gfs=false)

    β = 8.0
    μ = 0.0
    V = 0.25
    t_bethe = 1.0
    μ_bethe = 1.0

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    H = μ * op.n(1)
    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H, soi)

    # Hybridization propagator

    #Δ = V^2 * kd.ImaginaryTimeGF(kd.bethe_dos(t=t_bethe/2, ϵ=μ_bethe), grid)

    dlr = le.DLRGrid(Euv=1.25, β=β, isFermi=true, rtol=1e-12, rebuild=true, verbose=false)
    dlr_grid = DLRImaginaryTimeGrid(contour, dlr)
    Δ = DLRImaginaryTimeGF(kd.bethe_dos(t=t_bethe/2, ϵ=μ_bethe), dlr_grid)
    Δ.mat.data[:] = Δ.mat.data .* V^2
    
    # Pseudo Particle Strong Coupling Expansion

    if interpolate_gfs
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(ph_conj(Δ)))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=true)
    else
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), ph_conj(Δ))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])
    end

    ρ_0 = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P0), ed), ed)

    P_orders, P_orders_std = inchworm!(expansion,
                                       grid,
                                       orders,
                                       orders_bare,
                                       N_samples; n_pts_after_max=1,
                                       n_bare_steps=nτ-1)
                                       #n_bare_steps=4)

    if interpolate_gfs
        P = [p.GF for p in expansion.P]
        ppgf.normalize!(P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(P), ed), ed)
    else
        ppgf.normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P), ed), ed)

        ρ_wrm_orders = [
            full_hs_matrix(tofockbasis(ppgf.density_matrix(P_orders[o]), ed), ed)
            for o in sort(collect(keys(P_orders)))
        ]
        ρ_wrm_orders = [real(diag(r)) for r in ρ_wrm_orders]
        norm = sum(sum(ρ_wrm_orders))
        ρ_wrm_orders /= norm
        pto_hist = Array{Float64}([sum(r) for r in ρ_wrm_orders])
    end

    ρ_ref = diagm([1 - 0.5767879786180553, 0.5767879786180553])

    diff = maximum(abs.(ρ_ref - ρ_wrm))

    if ismaster()
        @show ρ_wrm_orders
        @show pto_hist
        @printf "ρ_0   = %16.16f %16.16f \n" real(diag(ρ_0))...
        @printf "ρ_ref = %16.16f %16.16f \n" real(diag(ρ_ref))...
        @printf "ρ_wrm = %16.16f %16.16f \n" real(diag(ρ_wrm))...
        @show diff
    end
    return diff, pto_hist
end

function run_nτ_calc(nτ, orders, N_sampless)

    orders_bare = orders

    diff_0, pto_hist_0 = run_bethe(nτ, orders, orders_bare, 0, interpolate_gfs=false)

    diffs_pto_hists = [
        run_bethe(nτ, orders, orders_bare,
                  N_samples, interpolate_gfs=false)
              for N_samples in N_sampless]

    diffs = [el[1] for el in diffs_pto_hists]
    pto_hists = [el[2] for el in diffs_pto_hists]

    if ismaster()
        max_order = maximum(orders)
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        filename = "data_bethe_ntau_$(nτ)_maxorder_$(max_order)_bary_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            g = h5.create_group(fid, "data")

            h5.attributes(g)["ntau"] = nτ
            h5.attributes(g)["diff_0"] = diff_0

            g["orders"] = collect(orders)
            g["orders_bare"] = collect(orders_bare)
            g["N_sampless"] = N_sampless

            g["diffs"] = diffs
            g["pto_hists"] = reduce(hcat, pto_hists)
        end
    end

end

nτs = [2]
#nτs = [64]
#nτs = [1024 * 8 * 4]
#N_sampless = 2 .^ (3:15)
#N_sampless = 2 .^ (3:15)

N_sampless = 2 .^ (3:18)

orderss = [0:5]
#orderss = [0:4]
#orderss = [0:3]

if ismaster()
    @show nτs
    @show N_sampless
    @show orderss
end

for orders in orderss
    for nτ in nτs
        run_nτ_calc(nτ, orders, N_sampless)
    end
end
