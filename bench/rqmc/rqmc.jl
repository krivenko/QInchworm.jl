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

using MD5
using ArgParse
using Random: MersenneTwister

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators

using QInchworm.utility: ph_conj
using QInchworm.ppgf: normalize!
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p
using QInchworm.randomization: RandomizationParams
using QInchworm.mpi: ismaster

function run_bethe(nτ, orders, orders_bare, orders_gf, N_samples, N_seqs, n_pts_after_max)

    β = 10.0
    μ = 0.0
    t_bethe = 2.0
    V = 0.5 * t_bethe
    μ_bethe = 0.0

    # ED solution

    H_imp = -μ * op.n(1)

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H_imp, soi)

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.bethe_dos(t=t_bethe/2, ϵ=μ_bethe), grid)

    # Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), ph_conj(Δ))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd])

    rand_params = RandomizationParams(MersenneTwister(12345678), N_seqs, .0)

    inchworm!(expansion, grid, orders, orders_bare, N_samples;
              n_pts_after_max=n_pts_after_max,
              rand_params=rand_params)

    λ = normalize!(expansion.P, β)

    P_std = sum(expansion.P_orders_std)
    # Normalize P_std using the same λ as for P
    for p in P_std
        normalize!(p, λ)
    end

    add_corr_operators!(expansion, (op.c(1), op.c_dag(1)))
    g, g_std = correlator_2p(expansion, grid, orders_gf, N_samples;
                             rand_params=rand_params)

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(g[1].mat.data...))))
        filename = "data_order_$(orders)_ntau_$(nτ)_N_samples_$(N_samples)_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            grp = h5.create_group(fid, "data")

            h5.attributes(grp)["beta"] = β
            h5.attributes(grp)["ntau"] = nτ
            h5.attributes(grp)["N_samples"] = N_samples
            h5.attributes(grp)["N_seqs"] = N_seqs
            h5.attributes(grp)["n_pts_after_max"] = n_pts_after_max

            grp["orders"] = collect(orders)
            grp["orders_bare"] = collect(orders_bare)
            grp["orders_gf"] = collect(orders_gf)

            grp["tau"] = collect(kd.imagtimes(g[1].grid))

            P_grp = h5.create_group(grp, "P")
            P_std_grp = h5.create_group(grp, "P_std")
            for n in axes(expansion.P, 1)
                P_grp[string(n)] = expansion.P[n].mat.data[1, 1, :]
                P_std_grp[string(n)] = P_std[n].mat.data[1, 1, :]
            end

            grp["gf"] = g[1].mat.data[1, 1, :]
            grp["gf_std"] = g_std[1].mat.data[1, 1, :]
            grp["gf_ref"] = -Δ.mat.data[1, 1, :] / V^2
        end
    end
end

s = ArgParseSettings()
@add_arg_table s begin
    "order"
        arg_type = Int
        help = "Maximal expansion order in the PPGF calculations"
    "ntau"
        arg_type = Int
        help = "Number of imaginary time slices in GF meshes"
    "N_samples"
        arg_type = Int
        help = "Number of qMC samples per Sobol sequence"
    "N_seqs"
        arg_type = Int
        default = 1
        help = "Number of scrambled Sobol sequences to be used"
    "--n_pts_after_max"
        arg_type = Int
        default = typemax(Int64)
        help = "Maximum number of points in the after-t_w region to be taken into account"
end

parsed_args = parse_args(ARGS, s)

order = parsed_args["order"]
nτ = parsed_args["ntau"]
N_samples = parsed_args["N_samples"]
N_seqs = parsed_args["N_seqs"]
n_pts_after_max = parsed_args["n_pts_after_max"]

if ismaster()
    println("order $(order) nτ $(nτ) N_samples $(N_samples) N_seqs $(N_seqs) " *
            "n_pts_after_max $(n_pts_after_max)")
end

order = parsed_args["order"]
order_gf = order - 1

orders = 0:order
orders_gf = 0:order_gf

run_bethe(nτ, orders, orders, orders_gf, N_samples, N_seqs, n_pts_after_max)
