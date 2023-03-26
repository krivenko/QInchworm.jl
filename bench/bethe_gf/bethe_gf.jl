""" Solve non-interacting two fermion AIM coupled to
semi-circular (Bethe lattice) hybridization functions.

Performing two kinds of tests:

1. Checking that the InchWorm expansion does not break particle-hole
symmetry for an AIM with ph-symmetry.

2. Compare to numerically exact results for the 1st, 2nd and 3rd
order dressed self-consistent expansion for the many-body
density matrix (computed using DLR elsewhere).

Note that the 1,2, 3 order density matrix differs from the
exact density matrix of the non-interacting system, since
the low order expansions introduce "artificial" effective
interactions between hybridization insertions.

Author: Hugo U. R. Strand (2023)

"""

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
using QInchworm.inchworm: inchworm_matsubara!, compute_gf_matsubara
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

function ρ_from_n_ref(ρ_wrm, n_ref)
    ρ_ref = zero(ρ_wrm)
    ρ_ref[1, 1] = (1 - n_ref) * (1 - n_ref)
    ρ_ref[2, 2] = n_ref * (1 - n_ref)
    ρ_ref[3, 3] = n_ref * (1 - n_ref)
    ρ_ref[4, 4] = n_ref * n_ref
    return ρ_ref
end

function ρ_from_ρ_ref(ρ_wrm, ρ_ref)
    ρ = zero(ρ_wrm)
    ρ[1, 1] = ρ_ref[1]
    ρ[2, 2] = ρ_ref[2]
    ρ[3, 3] = ρ_ref[3]
    ρ[4, 4] = ρ_ref[4]
    return ρ
end

function get_ρ_exact(ρ_wrm)
    n = 0.5460872495307262 # from DLR calc
    return ρ_from_n_ref(ρ_wrm, n)
end

function get_ρ_nca(ρ_wrm)
    rho_nca = [ 0.1961713995875524, 0.2474226001525296, 0.2474226001525296, 0.3089834001073883,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_nca)
end

function get_ρ_oca(ρ_wrm)
    rho_oca = [ 0.2018070389569783, 0.2476929924482211, 0.2476929924482211, 0.3028069761465793,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_oca)
end

function get_ρ_tca(ρ_wrm)
    rho_tca = [ 0.205163794520457, 0.2478638876741985, 0.2478638876741985, 0.2991084301311462,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_tca)
end

include("data_gf.jl")

function run_hubbard_dimer(ntau, orders, orders_bare, orders_gf, N_samples)

    #β = 1.0
    β = 10.0
    #V = 1.0
    V = 0.5
    #V = 0.2
    #V = 0.1
    μ = 0.0
    t_bethe = 1.0
    μ_bethe = 0.25

    # -- ED solution

    H_imp = -μ * (op.n(1) + op.n(2))

    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    # -- Hybridization propagator

    tau = [ real(im * τ.bpoint.val) for τ in grid ]
    delta_bethe = V^2 * semi_circular_g_tau(tau, t_bethe, μ_bethe, β)

    if true
        Δ = kd.ImaginaryTimeGF(
            (t1, t2) -> 1.0im * V^2 *
                semi_circular_g_tau(
                    [-imag(t1.bpoint.val - t2.bpoint.val)],
                    t_bethe, μ_bethe, β)[1],
            grid, 1, kd.fermionic, true)
    else
        Δ = kd.ImaginaryTimeGF(
            (t1, t2) -> -1.0im * 0.5 * V^2,
            grid, 1, kd.fermionic, true)
    end
    
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

    ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

    inchworm_matsubara!(expansion, grid, orders, orders_bare, N_samples)

    normalize!(expansion.P, β)
    ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(expansion.P), ed), ed)

    ρ_exa = get_ρ_exact(ρ_wrm)
    ρ_nca = get_ρ_nca(ρ_wrm)
    ρ_oca = get_ρ_oca(ρ_wrm)
    ρ_tca = get_ρ_tca(ρ_wrm)

    diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
    diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
    diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
    diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

    ρ_000 = real(diag(ρ_0))
    ρ_exa = real(diag(ρ_exa))
    ρ_nca = real(diag(ρ_nca))
    ρ_oca = real(diag(ρ_oca))
    ρ_tca = real(diag(ρ_tca))
    ρ_wrm = real(diag(ρ_wrm))

    if inch_print()
        @show ρ_000
        @show ρ_nca
        @show ρ_oca
        @show ρ_tca
        @show ρ_exa
        @show ρ_wrm

        @show sum(ρ_wrm)
        @show ρ_wrm[2] - ρ_wrm[3]

        @show diff_nca
        @show diff_oca
        @show diff_tca
        @show diff_exa
    end

    push!(expansion.corr_operators, (op.c(1), op.c_dag(1)))
    g = compute_gf_matsubara(expansion, grid, orders_gf, N_samples)

    if false
    diff_g_nca = maximum(abs.(get_g_nca() - g[1].mat.data[1, 1, :]))
    diff_g_oca = maximum(abs.(get_g_oca() - g[1].mat.data[1, 1, :]))
    diff_g_tca = maximum(abs.(get_g_tca() - g[1].mat.data[1, 1, :]))

    if inch_print()

        @show diff_g_nca
        @show diff_g_oca
        @show diff_g_tca

    end
    end

    if inch_print()

    τ = kd.imagtimes(g[1].grid)
    τ_ref = collect(LinRange(0, β, 128))

    plt.figure(figsize=(3.25*4, 12))
    subp = [5, 3, 1]

    for s in 1:length(expansion.P)
        plt.subplot(subp...); subp[end] += 1;

        x = collect(τ)
        y = collect(imag(expansion.P[s].mat.data[1, 1, :]))
        P_int = Interpolate(x, y)
        P = P_int.(τ_ref)
        
        #plt.plot(τ, -imag(expansion.P[s].mat.data[1, 1, :]), label="P$(s)")
        plt.plot(τ_ref, -P, label="P$(s)")
        plt.plot(τ_ref, -get_G_nca()[s], "k--", label="NCA $s ref", alpha=0.25)
        plt.plot(τ_ref, -get_G_oca()[s], "k:", label="OCA $s ref", alpha=0.25)
        plt.plot(τ_ref, -get_G_tca()[s], "k-.", label="TCA $s ref", alpha=0.25)
        plt.semilogy([], [])
        plt.ylabel(raw"$P_\Gamma(\tau)$")
        plt.xlabel(raw"$\tau$")
        plt.legend(loc="best")

        #P = imag(expansion.P[s].mat.data[1, 1, :])
        plt.subplot(subp...); subp[end] += 1;
        plt.plot(τ_ref, P - get_G_nca()[s], "k--", label="NCA $s ref", alpha=0.25)
        plt.plot(τ_ref, P - get_G_oca()[s], "k:", label="OCA $s ref", alpha=0.25)
        plt.plot(τ_ref, P - get_G_tca()[s], "k-.", label="TCA $s ref", alpha=0.25)
        plt.ylabel(raw"$\Delta P_\Gamma(\tau)$")
        plt.xlabel(raw"$\tau$")

        plt.subplot(subp...); subp[end] += 1;
        for (o, P) in enumerate(expansion.P_orders)
            p = imag(P[s].mat.data[1, 1, :])
            plt.semilogy(τ, -p, label="order $(o-1) ref", alpha=0.25)
        end
        plt.ylim([1e-9, 1e2])
        plt.ylabel(raw"$P_\Gamma(\tau)$")
        plt.xlabel(raw"$\tau$")
        plt.legend(loc="best")
    end

    plt.subplot(subp...); subp[end] += 1;
    plt.title("ntau = $(length(τ)), N_samples = $N_samples")
    plt.plot(τ_ref, imag(get_g_nca()), label="NCA ref")
    plt.plot(τ_ref, imag(get_g_oca()), label="OCA ref")
    plt.plot(τ_ref, imag(get_g_tca()), label="TCA ref")
    plt.plot(τ, imag(g[1].mat.data[1, 1, :]), "--", label="InchW")
    plt.xlabel(raw"$\tau$")
    plt.ylabel(raw"$G_{11}(\tau)$")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig("figure_ntau_$(ntau)_N_samples_$(N_samples)_orders_$(orders).pdf")
    #plt.show()

    end

    #return ρ_wrm, diff_exa, diff_nca, diff_oca, diff_tca, diff_g_nca, diff_g_oca
end

@testset "bethe_order_sweep" begin

    for o = [1, 2, 3]
        orders = 0:o
        orders_gf = 0:(o-1)        
        for ntau = [128]
            #for N_samples = [8*2^7]
            for N_samples = [8*2^6]
                run_hubbard_dimer(ntau, orders, orders, orders_gf, N_samples)
            end
        end
    end

end
