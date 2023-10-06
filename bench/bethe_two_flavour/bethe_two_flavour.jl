"""

Author: Hugo U. R. Strand (2023)

"""

using MPI; MPI.Init()
using HDF5; h5 = HDF5

using MD5
using ArgParse

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.utility
using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p
using QInchworm.mpi: ismaster

function run_bethe(nτ, orders, orders_bare, orders_gf, N_samples, n_pts_after_max)

    β = 10.0
    μ = 0.0
    t_bethe = 2.0
    V = 0.5 * t_bethe

    μ_bethe = 0.0
    B = 1.0

    # ED solution

    H_imp = -μ * (op.n(1) + op.n(2)) + B * op.n(1)

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.bethe_dos(ϵ=μ_bethe, t=t_bethe/2), grid)

    # Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))

    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))

    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    inchworm!(expansion, grid, orders, orders_bare, N_samples;
                        n_pts_after_max=n_pts_after_max)
    normalize!(expansion.P, β)

    add_corr_operators!(expansion, (op.c(1), op.c_dag(1))) # SPGF 1
    add_corr_operators!(expansion, (op.c(2), op.c_dag(2))) # SPGF 2
    add_corr_operators!(expansion, (op.c_dag(2)*op.c(1), op.c_dag(1)*op.c(2))) # <S+ S->
    add_corr_operators!(expansion, (op.c_dag(1)*op.c(1), op.c_dag(2)*op.c(2))) # <n_1 n_2>

    corr = correlator_2p(expansion, grid, orders_gf, N_samples)

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(corr[1].mat.data...))))
        filename = "data_order_$(orders)_ntau_$(nτ)_N_samples_$(N_samples)_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            grp = h5.create_group(fid, "data")

            h5.attributes(grp)["beta"] = β
            h5.attributes(grp)["ntau"] = nτ
            h5.attributes(grp)["n_pts_after_max"] = n_pts_after_max
            h5.attributes(grp)["N_samples"] = N_samples

            h5.attributes(grp)["B"] = B

            grp["orders"] = collect(orders)
            grp["orders_bare"] = collect(orders_bare)
            grp["orders_gf"] = collect(orders_gf)

            grp["tau"] = collect(kd.imagtimes(corr[1].grid))

            grp["g_1"] = corr[1].mat.data[1, 1, :]
            grp["g_2"] = corr[2].mat.data[1, 1, :]
            grp["SpSm"] = corr[3].mat.data[1, 1, :]
            grp["n1n2"] = corr[4].mat.data[1, 1, :]
            grp["delta"] = Δ.mat.data[1, 1, :]
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
