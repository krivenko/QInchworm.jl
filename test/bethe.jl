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

using MPI; MPI.Init()

using LinearAlgebra: diag

using QuadGK: quadgk

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.inchworm: inchworm!
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

function run_hubbard_dimer(ntau, orders, orders_bare, N_samples, μ_bethe)

    β = 10.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0

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
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

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

    return ρ_wrm, diff_exa, diff_nca, diff_oca, diff_tca
end

@testset "bethe_ph_symmetry" begin

    ntau = 3
    N_samples = 2^4
    μ_bethe = 0.0

    tests = [
        (0:0, 0:0), # ok
        (0:1, 0:0), # ok
        (0:0, 0:1), # ok
        (0:2, 0:0), # ok
        (0:0, 0:2), # ok
        (0:3, 0:0), # ok
        (0:0, 0:3), # ok
        #(0:4, 0:0), # ok, but too slow for testing
        #(0:0, 0:4), # ok, but too slow for testing
        ]

    for (orders_bare, orders) in tests
        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
            run_hubbard_dimer(ntau, orders, orders_bare, N_samples, μ_bethe)
        @show orders_bare, orders
        @test ρ ≈ [0.25, 0.25, 0.25, 0.25]
    end

end

@testset "bethe_order1" begin

    ntau = 128
    orders = 0:1
    N_samples = 8 * 2^5
    μ_bethe = 0.25

    ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
        run_hubbard_dimer(ntau, orders, orders, N_samples, μ_bethe)

    @test diffs_nca < 2e-3
    @test diffs_nca < diffs_oca
    @test diffs_nca < diffs_tca
    @test diffs_nca < diffs_exa

end

@testset "bethe_order2" begin

    ntau = 128
    orders = 0:2
    N_samples = 8 * 2^5
    μ_bethe = 0.25

    ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
        run_hubbard_dimer(ntau, orders, orders, N_samples, μ_bethe)

    @test diffs_oca < 4e-3
    @test diffs_oca < diffs_nca
    @test diffs_oca < diffs_tca
    @test diffs_oca < diffs_exa

end

@testset "bethe_order3" begin

    ntau = 128
    orders = 0:3
    N_samples = 8 * 2^5
    μ_bethe = 0.25

    ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
        run_hubbard_dimer(ntau, orders, orders, N_samples, μ_bethe)

    @test diffs_tca < 4e-3
    @test diffs_tca < diffs_nca
    @test diffs_tca < diffs_oca
    #@test diffs_tca < diffs_exa

end
