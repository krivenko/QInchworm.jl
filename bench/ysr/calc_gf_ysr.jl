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
# Authors: Igor Krivenko

# This script reproduces the numerical simulation setup used in
#
#   Xia, HN., Minamitani, E., Žitko, R. et al.
#   Spin-orbital Yu-Shiba-Rusinov states in single Kondo molecular magnet
#   Nat Commun 13, 6388 (2022)
#   https://doi.org/10.1038/s41467-022-34187-8
#

using MPI; MPI.Init()
using HDF5; h5 = HDF5

using ArgParse
using Random: MersenneTwister

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

include("dos.jl")

using QInchworm.utility: ph_conj
using QInchworm.ppgf: normalize!
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: diff_inchworm!, correlator_2p
using QInchworm.randomization: RandomizationParams, RequestStdDev
using QInchworm.mpi: ismaster

function calc_gf_yrs(ϵ, U, Δ, J, D, Γ, β, orders, orders_gf, nτ, N_samples, N_seqs)

    ## ED solution
    soi = ked.Hilbert.SetOfIndices([[s, o] for s in ("up", "dn") for o in 1:2])

    # Orbital energies
    H_imp = sum(ϵ[o] * op.n(s, o) for s in ("up", "dn") for o in 1:2)
    # Coulomb repulsion
    H_imp += sum(U[o] * op.n(s, o) for s in ("up", "dn") for o in 1:2)

    S_z = [0.5 * (op.n("up", o) - op.n("dn", o)) for o in 1:2]
    S_p = [op.c_dag("up", o) * op.c("dn", o) for o in 1:2]
    S_m = [op.c_dag("dn", o) * op.c("up", o) for o in 1:2]

    # Hund's coupling
    H_imp += -J * (S_z[1] * S_z[2] + 0.5 * (S_p[1] * S_m[2] + S_m[1] * S_p[2]))

    ed = ked.EDCore(H_imp, soi)

    # Pseudo Particle Strong Coupling Expansion

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    ips = Array{InteractionPair{kd.ImaginaryTimeGF{ComplexF64, true}}, 1}()

    # Hybridization function
    for o in 1:2
        V2 = Γ[o] / π * (2D)

        # Normal component
        Δ_normal = V2 * kd.ImaginaryTimeGF(normal_dos(D=D, Δ=Δ), grid)
        for s in ("up", "dn")
            push!(ips, InteractionPair(op.c_dag(s, o), op.c(s, o), Δ_normal))
            push!(ips, InteractionPair(op.c(s, o), op.c_dag(s, o), ph_conj(Δ_normal)))
        end
        # Anomalous component
        Δ_anomalous = -V2 * kd.ImaginaryTimeGF(anomalous_dos(D=D, Δ=Δ), grid)
        push!(ips, InteractionPair(op.c_dag("up", o), op.c_dag("dn", o), Δ_anomalous))
        push!(ips, InteractionPair(op.c_dag("dn", o), op.c_dag("up", o), -Δ_anomalous))
        push!(ips, InteractionPair(op.c("dn", o), op.c("up", o), Δ_anomalous))
        push!(ips, InteractionPair(op.c("up", o), op.c("dn", o), -Δ_anomalous))
    end

    expansion = Expansion(ed, grid, ips)

    rand_params = RandomizationParams(MersenneTwister(12345678), N_seqs, .0)

    diff_inchworm!(expansion, grid, orders, N_samples; rand_params=rand_params)
    normalize!(expansion.P, β)

    for s in ("up", "dn"), o in 1:2
        add_corr_operators!(expansion, (op.c(s, o), op.c_dag(s, o)))
    end

    g, g_std = correlator_2p(expansion, grid, orders_gf, N_samples, RequestStdDev();
                             rand_params=rand_params)

    return expansion, g, g_std
end

s = ArgParseSettings()
@add_arg_table s begin
    "order"
        arg_type = Int
        required = true
        help = "Highest expansion order to account for"
    "ntau"
        arg_type = Int
        required = true
        help = "Number of imaginary time slices"
    "N_samples"
        arg_type = Int
        required = true
        help = "Number of qMC samples to be taken"
    "N_seqs"
        arg_type = Int
        default = 4
        help = "Number of scrambled Sobol sequences to be used"
    "--eps1"
        arg_type = Float64
        default = -25.0
        help = "Energy ε₁ of the first ligand orbital of Tb₂Pc₃"
    "--eps2"
        arg_type = Float64
        default = -4.0
        help = "Energy ε₂ of the second ligand orbital of Tb₂Pc₃"
    "--U1"
        arg_type = Float64
        default = 100.0
        help = "Coulomb constant U₁ of the first ligand orbital of Tb₂Pc₃"
    "--U2"
        arg_type = Float64
        default = 10.0
        help = "Coulomb constant U₂ of the second ligand orbital of Tb₂Pc₃"
    "--Delta"
        arg_type = Float64
        default = 1.0
        help = "Superconducting gap Δ of the Pb(111) substrate"
    "--J"
        arg_type = Float64
        default = -0.5
        help = "Hund's coupling constant"
    "--D"
        arg_type = Float64
        default = 10.0
        help = "Half-bandwidth of the normal state spectrum of the Pb(111) substrate"
    "--Gamma1"
        arg_type = Float64
        default = 8.0
        help =
           "Hybridization strength Γ₁ between the first ligand orbital and the substrate"
    "--Gamma2"
        arg_type = Float64
        default = 3.75
        help =
            "Hybridization strength Γ₂ between the second ligand orbital and the substrate"
    "--beta"
        arg_type = Float64
        default = 10.0
        help = "Inverse temperature β"
end

parsed_args = parse_args(ARGS, s)

order = parsed_args["order"]
orders = 0:order
orders_gf = 0:(order - 1)

nτ = parsed_args["ntau"]
N_samples = parsed_args["N_samples"]
N_seqs = parsed_args["N_seqs"]

exp, g, g_std = calc_gf_yrs(
    [parsed_args["eps1"], parsed_args["eps2"]],
    [parsed_args["U1"], parsed_args["U2"]],
    parsed_args["Delta"],
    parsed_args["J"],
    parsed_args["D"],
    [parsed_args["Gamma1"], parsed_args["Gamma2"]],
    parsed_args["beta"],
    orders,
    0:(order - 1),
    nτ,
    N_samples,
    N_seqs
)

if ismaster()
	h5.h5open("gf_ysr.order$(order)_ntau$(nτ)_N_samples$(N_samples)_N_seqs$(N_seqs).h5", "w") do fid
        grp = h5.create_group(fid, "data")

        h5.attributes(grp)["eps1"] = parsed_args["eps1"]
        h5.attributes(grp)["eps2"] = parsed_args["eps2"]
        h5.attributes(grp)["U1"] = parsed_args["U1"]
        h5.attributes(grp)["U2"] = parsed_args["U2"]
        h5.attributes(grp)["Delta"] = parsed_args["Delta"]
        h5.attributes(grp)["J"] = parsed_args["J"]
        h5.attributes(grp)["D"] = parsed_args["D"]
        h5.attributes(grp)["Gamma1"] = parsed_args["Gamma1"]
        h5.attributes(grp)["Gamma2"] = parsed_args["Gamma2"]
        h5.attributes(grp)["beta"] = parsed_args["beta"]
        h5.attributes(grp)["N_samples"] = parsed_args["N_samples"]
        h5.attributes(grp)["N_seqs"] = parsed_args["N_seqs"]

        grp["orders"] = collect(orders)
        grp["orders_gf"] = collect(orders_gf)

        grp["tau"] = collect(kd.imagtimes(first(g).grid))

        grp_gf = h5.create_group(grp, "gf")
        grp_gf_std = h5.create_group(grp, "gf_std")

        for (i, el) in enumerate(["up_11", "up_22", "dn_11", "dn_22"])
	    grp_gf[el] = g[i].mat.data[1, 1, :]
	    grp_gf_std[el] = g_std[i].mat.data[1, 1, :]
        end

        for (s, p) in enumerate(exp.P)
            grp["P_$(s)"] = p.mat.data
            grp["P0_$(s)"] = exp.P0[s].mat.data
        end
    end
end
