"""

Author: Hugo U. R. Strand (2023)

"""

using MD5
using HDF5; h5 = HDF5

using Test
using LinearInterpolations: Interpolate

using MPI; MPI.Init()

import PyPlot; plt = PyPlot

using LinearAlgebra: diag
using QuadGK: quadgk

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p
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

function run_bethe(ntau, orders, orders_bare, orders_gf, N_samples, n_pts_after_max)

    β = 10.0
    μ = 0.0
    t_bethe = 2.0
    V = 0.5 * t_bethe

    μ_bethe = 0.0
    B = 1.0

    # -- ED solution

    H_imp = -μ * (op.n(1) + op.n(2)) + B * op.n(1)

    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    # -- Hybridization propagator

    tau = [ real(im * τ.bpoint.val) for τ in grid ]

    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> 1.0im * V^2 *
            semi_circular_g_tau([-imag(t1.bpoint.val - t2.bpoint.val)], t_bethe, μ_bethe, β)[1],
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
    
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))

    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    inchworm!(expansion, grid, orders, orders_bare, N_samples;
                        n_pts_after_max=n_pts_after_max)
    normalize!(expansion.P, β)

    add_corr_operators!(expansion, (op.c(1), op.c_dag(1))) # spgf 1
    add_corr_operators!(expansion, (op.c(2), op.c_dag(2))) # spgf 2
    add_corr_operators!(expansion, (op.c_dag(2)*op.c(1), op.c_dag(1)*op.c(2))) # <S+ S->
    add_corr_operators!(expansion, (op.c_dag(1)*op.c(1), op.c_dag(2)*op.c(2))) # <n_1 n_2>

    g = correlator_2p(expansion, grid, orders_gf, N_samples)

    # ==
    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(g[1].mat.data...))))
        filename = "data_order_$(orders)_ntau_$(ntau)_N_samples_$(N_samples)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")
        grp = h5.create_group(fid, "data")

        h5.attributes(grp)["beta"] = β
        h5.attributes(grp)["ntau"] = ntau
        h5.attributes(grp)["n_pts_after_max"] = n_pts_after_max
        h5.attributes(grp)["N_samples"] = N_samples

        h5.attributes(grp)["B"] = B
        
        grp["orders"] = collect(orders)
        grp["orders_bare"] = collect(orders_bare)
        grp["orders_gf"] = collect(orders_gf)

        grp["tau"] = collect(kd.imagtimes(g[1].grid))

        grp["g_1"] = g[1].mat.data[1, 1, :]
        grp["g_2"] = g[2].mat.data[1, 1, :]
        grp["SpSm"] = g[3].mat.data[1, 1, :]
        grp["n1n2"] = g[4].mat.data[1, 1, :]
        grp["delta"] = Δ.mat.data[1, 1, :]

        h5.close(fid)
    end
end



@assert length(ARGS) == 4

order = parse(Int, ARGS[1])
ntau = parse(Int, ARGS[2])
N_samples = parse(Int, ARGS[3])
n_pts_after_max = parse(Int, ARGS[4])

if n_pts_after_max == 0
    n_pts_after_max = typemax(Int64)
end

order_gf = order - 1

if ismaster()
    println("order $(order) ntau $(ntau) N_samples $(N_samples) n_pts_after_max $(n_pts_after_max)")
end

orders = 0:order
orders_gf = 0:order_gf

run_bethe(ntau, orders, orders, orders_gf, N_samples, n_pts_after_max)
