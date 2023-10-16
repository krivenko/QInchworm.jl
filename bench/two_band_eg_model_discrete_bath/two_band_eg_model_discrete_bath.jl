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

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.utility: ph_conj
using QInchworm.ppgf: normalize!, density_matrix, atomic_ppgf!
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p
using QInchworm.mpi: ismaster

function make_hamiltonian(n_orb, mu, U, J)
    soi = ked.Hilbert.SetOfIndices([[s,o] for s in ("up", "dn") for o in 1:n_orb])

    H = op.OperatorExpr{Float64}()

    H += -mu * sum((op.n("up", o) + op.n("dn", o)) for o in 1:n_orb)
    H += U * sum(op.n("up", o) * op.n("dn", o) for o in 1:n_orb)

    H += (U - 2 * J) * sum(op.n("up", o1) * op.n("dn", o2)
                           for o1 in 1:n_orb, o2 in 1:n_orb if o1 != o2)

    H += (U - 3 * J) * sum(op.n(s, o1) * op.n(s, o2)
                           for s in ("up", "dn"), o1 in 1:n_orb, o2 in 1:n_orb if o2 < o1)

    H += -J * sum(op.c_dag("up", o1) * op.c_dag("dn", o1) * op.c("up", o2) * op.c("dn", o2)
                  for o1 in 1:n_orb, o2 in 1:n_orb if o1 != o2)
    H += -J * sum(op.c_dag("up", o1) * op.c_dag("dn", o2) * op.c("up", o2) * op.c("dn", o1)
                  for o1 in 1:n_orb, o2 in 1:n_orb if o1 != o2)

    return (soi, H)
end

function run_bethe(nτ, orders, orders_bare, orders_gf, N_samples, n_pts_after_max;
                   discrete_bath=true)

    n_orb = 2
    β = 8.0

    t_bethe = 2.0
    μ_bethe = 0.0

    e_k = 2.3

    U = 2.0
    J = 0.2
    μ = (3*U - 5*J)/2 - 1.5

    # ED solution
    soi, H_imp = make_hamiltonian(n_orb, μ, U, J)

    # Need to break symmetry of ED!

    symm_breakers = [
        op.c_dag("up", 1) * op.c("up", 2) + op.c_dag("up", 2) * op.c("up", 1),
        op.c_dag("dn", 1) * op.c("dn", 2) + op.c_dag("dn", 2) * op.c("dn", 1)
    ]

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)
    ed = ked.EDCore(H_imp, soi, symmetry_breakers=symm_breakers)

    # Hybridization propagator

    if discrete_bath
        ismaster() && println("--> Discrete Bath")

        Δ = kd.ImaginaryTimeGF(kd.DeltaDOS([+e_k, -e_k], [1.0, 1.0]), grid)
    else
        ismaster() && println("--> Bethe Bath")

        Δ = kd.ImaginaryTimeGF(kd.bethe_dos(t=t_bethe, ϵ=μ_bethe), grid)
    end

    # Pseudo Particle Strong Coupling Expansion

    ips = Array{InteractionPair{kd.ImaginaryTimeGF{ComplexF64, true}}, 1}()

    for s in ["up", "dn"], o in [1, 2]
        push!(ips, InteractionPair(op.c_dag(s, o), op.c(s, o), Δ))
        push!(ips, InteractionPair(op.c(s, o), op.c_dag(s, o), ph_conj(Δ)))
    end

    for s in ["up", "dn"]
        push!(ips, InteractionPair(op.c_dag(s, 1), op.c(s, 2), Δ))
        push!(ips, InteractionPair(op.c(s, 2), op.c_dag(s, 1), ph_conj(Δ)))

        push!(ips, InteractionPair(op.c_dag(s, 2), op.c(s, 1), Δ))
        push!(ips, InteractionPair(op.c(s, 1), op.c_dag(s, 2), ph_conj(Δ)))
    end

    expansion = Expansion(ed, grid, ips)
    atomic_ppgf!(expansion.P0, ed, Δλ=2.0)

    inchworm!(expansion, grid, orders, orders_bare, N_samples;
              n_pts_after_max=n_pts_after_max)

    P_raw = deepcopy(expansion.P)
    normalize!(expansion.P, β)

    add_corr_operators!(expansion, (op.c("up", 1), op.c_dag("up", 1)))
    add_corr_operators!(expansion, (op.c("dn", 1), op.c_dag("dn", 1)))

    add_corr_operators!(expansion, (op.c("up", 2), op.c_dag("up", 2)))
    add_corr_operators!(expansion, (op.c("dn", 2), op.c_dag("dn", 2)))

    add_corr_operators!(expansion, (op.c("up", 1), op.c_dag("up", 2)))
    add_corr_operators!(expansion, (op.c("dn", 1), op.c_dag("dn", 2)))

    add_corr_operators!(expansion, (op.c("up", 2), op.c_dag("up", 1)))
    add_corr_operators!(expansion, (op.c("dn", 2), op.c_dag("dn", 1)))

    g = correlator_2p(expansion, grid, orders_gf, N_samples)

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(g[1].mat.data...))))
        filename = "data_order_$(orders)_ntau_$(nτ)_N_samples_$(N_samples)_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            grp = h5.create_group(fid, "data")

            h5.attributes(grp)["beta"] = β
            h5.attributes(grp)["ntau"] = nτ
            h5.attributes(grp)["n_pts_after_max"] = n_pts_after_max
            h5.attributes(grp)["N_samples"] = N_samples

            grp["orders"] = collect(orders)
            grp["orders_bare"] = collect(orders_bare)
            grp["orders_gf"] = collect(orders_gf)

            grp["tau"] = collect(kd.imagtimes(first(g).grid))

            grp["gf_up_11"] = g[1].mat.data[1, 1, :]
            grp["gf_dn_11"] = g[2].mat.data[1, 1, :]

            grp["gf_up_22"] = g[3].mat.data[1, 1, :]
            grp["gf_dn_22"] = g[4].mat.data[1, 1, :]

            grp["gf_up_12"] = g[5].mat.data[1, 1, :]
            grp["gf_dn_12"] = g[6].mat.data[1, 1, :]

            grp["gf_up_21"] = g[7].mat.data[1, 1, :]
            grp["gf_dn_21"] = g[8].mat.data[1, 1, :]

            for (s, p) in enumerate(expansion.P)
                grp["P_$(s)"] = p.mat.data
                grp["P0_$(s)"] = expansion.P0[s].mat.data
                grp["Praw_$(s)"] = P_raw[s].mat.data
            end

            grp["gf_ref"] = -Δ.mat.data[1, 1, :]
        end
    end
end

s = ArgParseSettings()
@add_arg_table s begin
    "order"
        help = "Highest expansion order to account for"
        arg_type = Int
    "ntau"
        help = "Number of imaginary time slices"
        arg_type = Int
    "N_samples"
        help = "Number of qMC samples to be taken"
        arg_type = Int
    "--n_pts_after_max"
        help = "Maximal number of points in the after-t_w region"
        arg_type = Int
        default = typemax(Int64)
end

parsed_args = parse_args(ARGS, s)

order = parsed_args["order"]
nτ = parsed_args["ntau"]
N_samples = parsed_args["N_samples"]
n_pts_after_max = parsed_args["n_pts_after_max"]

if ismaster()
    println("order $(order) nτ $(nτ) N_samples $(N_samples) " *
            "n_pts_after_max $(n_pts_after_max)")
end

orders = 0:order
orders_gf = 0:(order - 1)

run_bethe(nτ, orders, orders, orders_gf, N_samples, n_pts_after_max)
