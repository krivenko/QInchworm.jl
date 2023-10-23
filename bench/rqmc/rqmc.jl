"""

Author: Hugo U. R. Strand (2023)

"""

using ArgParse
using MD5
using QuadGK: quadgk
using Random: MersenneTwister
using HDF5; h5 = HDF5

using MPI; MPI.Init()


using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: partition_function, normalize!
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p
using QInchworm.randomization: RandomizationParams
using QInchworm.mpi: ismaster

function semi_circular_g_tau(times, t, h, β)

    g_out = zero(times)

    function kernel(t, w)
        if w > 0
            return exp(-t * w) / (1 + exp(-w))
        else
            return exp((1 - t)*w) / (1 + exp(w))
        end
    end

    for (i, τ) in enumerate(times)
        I = x -> -2 / pi / t^2 * kernel(τ/β, β*x) * sqrt(x + t - h) * sqrt(t + h - x)
        g, err = quadgk(I, -t+h, t+h; rtol=1e-12)
        g_out[i] = g
    end

    return g_out
end

function run_bethe(ntau, orders, orders_bare, orders_gf, N_samples, N_seqs, n_pts_after_max)

    β = 10.0
    μ = 0.0
    t_bethe = 2.0
    V = 0.5 * t_bethe
    μ_bethe = 0.0

    # -- ED solution

    H_imp = -μ * op.n(1)

    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H_imp, soi)

    # -- Hybridization propagator

    tau = [ real(im * τ.bpoint.val) for τ in grid ]
    delta_bethe = V^2 * semi_circular_g_tau(tau, t_bethe, μ_bethe, β)

    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> 1.0im * V^2 *
            semi_circular_g_tau(
                [-imag(t1.bpoint.val - t2.bpoint.val)],
                t_bethe, μ_bethe, β)[1],
        grid, 1, kd.fermionic, true)

    function reverse(g::kd.ImaginaryTimeGF)
        g_rev = deepcopy(g)
        τ_0, τ_β = first(g.grid), last(g.grid)
        for τ in g.grid
            g_rev[τ, τ_0] = g[τ_β, τ]
        end
        return g_rev
    end

    # -- Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
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

    # ==
    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(g[1].mat.data...))))
        filename = "data_order_$(orders)_ntau_$(ntau)_N_samples_$(N_samples)_N_seqs_$(N_seqs)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")
        grp = h5.create_group(fid, "data")

        h5.attributes(grp)["beta"] = β
        h5.attributes(grp)["ntau"] = ntau
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

        h5.close(fid)
    end
end

s = ArgParseSettings()
@add_arg_table s begin
    "--order"
        arg_type = Int
        help = "Maximal expansion order in the PPGF calculations"
    "--ntau"
        arg_type = Int
        help = "Number of imaginary time slices in GF meshes"
    "--n_pts_after_max"
        arg_type = Int
        default = typemax(Int64)
        help = "Maximum number of points in the after-t_w region to be taken into account"
    "--N_samples"
        arg_type = Int
        help = "Number of qMC samples per Sobol sequence"
    "--N_seqs"
        arg_type = Int
        default = 1
        help = "Number of scrambled Sobol sequences to be used"
end

parsed_args = parse_args(ARGS, s)
if ismaster()
    for (arg, val) in parsed_args
        println("$arg => $val")
    end
end

order = parsed_args["order"]
order_gf = order - 1

orders = 0:order
orders_gf = 0:order_gf

run_bethe(parsed_args["ntau"],
          orders,
          orders,
          orders_gf,
          parsed_args["N_samples"],
          parsed_args["N_seqs"],
          parsed_args["n_pts_after_max"])
