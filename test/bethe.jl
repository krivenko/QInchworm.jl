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

using MPI; MPI.Init()
using HDF5

using LinearAlgebra: diag, tr
using QuadGK: quadgk

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.inchworm: inchworm!
using QInchworm.utility: inch_print

# TODO: Use dos2gf() from Keldysh.jl
function semi_circular_g_tau(τ, t, h, β)

    function kernel(t, w)
        if w > 0
            return exp(-t * w) / (1 + exp(-w))
        else
            return exp((1 - t)*w) / (1 + exp(w))
        end
    end

    I = x -> -2 / pi / t^2 * kernel(τ/β, β*x) * sqrt(x + t - h) * sqrt(t + h - x)
    g, err = quadgk(I, -t+h, t+h; rtol=1e-12)
    return g
end

@testset "bethe" begin

    # -- Reference results from DLR calculations

    ref_fid = HDF5.h5open((@__DIR__) * "/bethe.h5", "r")

    ρ_exa = HDF5.read(ref_fid["/rho/exact"])
    ρ_nca = HDF5.read(ref_fid["/rho/NCA"])
    ρ_oca = HDF5.read(ref_fid["/rho/OCA"])
    ρ_tca = HDF5.read(ref_fid["/rho/TCA"])

    HDF5.close(ref_fid)

    function run_bethe(nτ, orders, orders_bare, N_samples, μ_bethe)

        β = 10.0
        V = 0.5
        μ = 0.0
        t_bethe = 1.0

        # -- ED solution

        H_imp = -μ * (op.n(1) + op.n(2))

        # -- Impurity problem

        contour = kd.ImaginaryContour(β=β);
        grid = kd.ImaginaryTimeGrid(contour, nτ);

        soi = ked.Hilbert.SetOfIndices([[1], [2]])
        ed = ked.EDCore(H_imp, soi)

        # -- Hybridization propagator

        Δ = kd.ImaginaryTimeGF(
            (t1, t2) -> 1.0im * V^2 *
                semi_circular_g_tau(
                    -imag(t1.bpoint.val - t2.bpoint.val),
                    t_bethe, μ_bethe, β)[1],
            grid, 1, kd.fermionic, true)

        # -- Pseudo Particle Strong Coupling Expansion

        function reverse(g::kd.ImaginaryTimeGF)
            g_rev = deepcopy(g)
            τ_0, τ_β = first(g.grid), last(g.grid)
            for τ in g.grid
                g_rev[τ, τ_0] = g[τ_β, τ]
            end
            return g_rev
        end

        ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
        ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
        ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))
        expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

        ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

        inchworm!(expansion, grid, orders, orders_bare, N_samples)

        normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(expansion.P), ed), ed)

        diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
        diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
        diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
        diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

        if inch_print()
            @show real(diag(ρ_0))
            @show real(diag(ρ_nca))
            @show real(diag(ρ_oca))
            @show real(diag(ρ_tca))
            @show real(diag(ρ_exa))
            @show real(diag(ρ_wrm))

            @show tr(ρ_wrm)
            @show ρ_wrm[2, 2] - ρ_wrm[3, 3]

            @show diff_nca
            @show diff_oca
            @show diff_tca
            @show diff_exa
        end

        return real(diag(ρ_wrm)), diff_exa, diff_nca, diff_oca, diff_tca
    end

    @testset "ph_symmetry" begin
        nτ = 3
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
            @show orders_bare, orders
            ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
                run_bethe(nτ, orders, orders_bare, N_samples, μ_bethe)
            @test ρ ≈ [0.25, 0.25, 0.25, 0.25]
        end
    end

    @testset "order1" begin
        nτ = 128
        orders = 0:1
        N_samples = 8 * 2^5
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
            run_bethe(nτ, orders, orders, N_samples, μ_bethe)

        @test diffs_nca < 2e-3
        @test diffs_nca < diffs_oca
        @test diffs_nca < diffs_tca
        @test diffs_nca < diffs_exa
    end

    @testset "order2" begin
        nτ = 128
        orders = 0:2
        N_samples = 8 * 2^5
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
            run_bethe(nτ, orders, orders, N_samples, μ_bethe)

        @test diffs_oca < 4e-3
        @test diffs_oca < diffs_nca
        @test diffs_oca < diffs_tca
        @test diffs_oca < diffs_exa
    end

    @testset "order3" begin
        nτ = 128
        orders = 0:3
        N_samples = 8 * 2^5
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
            run_bethe(nτ, orders, orders, N_samples, μ_bethe)

        @test diffs_tca < 4e-3
        @test diffs_tca < diffs_nca
        @test diffs_tca < diffs_oca
        #@test diffs_tca < diffs_exa
    end

end
