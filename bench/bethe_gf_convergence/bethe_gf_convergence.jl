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
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.inchworm: inchworm_matsubara!, correlator_2p
using QInchworm.utility: inch_print

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

    inchworm_matsubara!(expansion, grid, orders, orders_bare, N_samples;
                        n_pts_after_max=n_pts_after_max)
    normalize!(expansion.P, β)

    push!(expansion.corr_operators, (op.c(1), op.c_dag(1)))
    g = correlator_2p(expansion, grid, orders_gf, N_samples)

    # ==
    if inch_print()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(g[1].mat.data...))))
        filename = "data_order_$(orders)_ntau_$(ntau)_N_samples_$(N_samples)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")
        grp = h5.create_group(fid, "data")

        h5.attributes(grp)["beta"] = β
        h5.attributes(grp)["ntau"] = ntau
        h5.attributes(grp)["n_pts_after_max"] = n_pts_after_max
        h5.attributes(grp)["N_samples"] = N_samples

        grp["orders"] = collect(orders)
        grp["orders_bare"] = collect(orders_bare)
        grp["orders_gf"] = collect(orders_gf)

        grp["tau"] = collect(kd.imagtimes(g[1].grid))
        grp["gf"] = g[1].mat.data[1, 1, :]
        grp["gf_ref"] = -Δ.mat.data[1, 1, :] / V^2

        h5.close(fid)
    end

    #if inch_print()
    if false

        τ = kd.imagtimes(g[1].grid)
        τ_ref = collect(LinRange(0, β, 128))

        plt.figure(figsize=(3.25*2, 8))
        subp = [3, 2, 1]

        for s in 1:length(expansion.P)
            plt.subplot(subp...); subp[end] += 1;

            x = collect(τ)
            y = collect(imag(expansion.P[s].mat.data[1, 1, :]))
            P_int = Interpolate(x, y)
            P = P_int.(τ_ref)

            plt.plot(τ_ref, -P, label="P$(s)")
            plt.semilogy([], [])
            plt.ylabel(raw"$P_\Gamma(\tau)$")
            plt.xlabel(raw"$\tau$")
            plt.legend(loc="best")

            plt.subplot(subp...); subp[end] += 1;
            for (o, P) in enumerate(expansion.P_orders)
                p = imag(P[s].mat.data[1, 1, :])
                plt.semilogy(τ, -p, label="order $(o-1) ref", alpha=0.25)
            end
            plt.ylim(bottom=1e-9)
            plt.ylabel(raw"$P_\Gamma(\tau)$")
            plt.xlabel(raw"$\tau$")
            plt.legend(loc="best")
        end

        x = collect(τ)
        y = collect(imag(g[1].mat.data[1, 1, :]))
        g_int = Interpolate(x, y)
        gr = g_int.(τ_ref)

        plt.subplot(subp...); subp[end] += 1;
        plt.title("ntau = $(length(τ)), N_samples = $N_samples")
        plt.plot(τ, imag(g[1].mat.data[1, 1, :]), "--", label="InchW")
        plt.plot(τ, -imag(Δ.mat.data[1, 1, :])/V^2, "--", label="Bethe")
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$G_{11}(τ)$")
        plt.legend(loc="best")
        plt.ylim(bottom=0)

        plt.tight_layout()
        plt.savefig("figure_ntau_$(ntau)_N_samples_$(N_samples)_orders_$(orders).pdf")
        #plt.show()
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

if inch_print()
    println("order $(order) ntau $(ntau) N_samples $(N_samples) n_pts_after_max $(n_pts_after_max)")
end

#exit()

orders = 0:order
orders_gf = 0:order_gf

run_bethe(ntau, orders, orders, orders_gf, N_samples, n_pts_after_max)

exit()

# -- Old sweeps

n_pts_after_max = 1
#n_pts_after_max = typemax(Int64)

#for o = [1, 2, 3, 4, 5]
#for o = [4, 5]
for o = [7]
    #for o = [6, 7, 8]
    orders = 0:o
    orders_gf = 0:(o-1)
    #for ntau = [64, 128, 256]
    for ntau = [16]
        #for N_samples = 32 * 2 .^ range(0, 10)
        for N_samples = [8*2^0]
            run_bethe(ntau, orders, orders, orders_gf, N_samples, n_pts_after_max)
        end
    end
end
