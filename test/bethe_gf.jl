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

using Test

using MPI; MPI.Init()
using HDF5

using LinearAlgebra: diag, tr, I

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p

# Solve non-interacting two fermion AIM coupled to
# semi-circular (Bethe lattice) hybridization functions.
#
# Performing two kinds of tests:
#
# 1. Checking that the InchWorm expansion does not break particle-hole
# symmetry for an AIM with ph-symmetry.
#
# 2. Compare to numerically exact results for the 1st, 2nd and 3rd
# order dressed self-consistent expansion for the many-body
# density matrix (computed using DLR elsewhere).
#
# Note that the 1, 2, 3 order density matrix differs from the
# exact density matrix of the non-interacting system, since
# the low order expansions introduce "artificial" effective
# interactions between hybridization insertions.
@testset "Bethe GF" verbose=true begin

    # Reference results from DLR calculations

    ref_fid = HDF5.h5open((@__DIR__) * "/bethe.h5", "r")

    ρ_exa = HDF5.read(ref_fid["/rho/exact"])
    ρ_nca = HDF5.read(ref_fid["/rho/NCA"])
    ρ_oca = HDF5.read(ref_fid["/rho/OCA"])
    ρ_tca = HDF5.read(ref_fid["/rho/TCA"])

    g_nca = HDF5.read(ref_fid["/g/NCA"])
    g_oca = HDF5.read(ref_fid["/g/OCA"])
    g_tca = HDF5.read(ref_fid["/g/TCA"])

    HDF5.close(ref_fid)

    function run_bethe_gf(nτ, orders, orders_bare, orders_gf, N_samples, μ_bethe)

        β = 10.0
        V = 0.5
        μ = 0.0
        t_bethe = 0.5

        # Hamiltonian

        H_imp = -μ * (op.n(1) + op.n(2))
        soi = ked.Hilbert.SetOfIndices([[1], [2]])

        # Imaginary time grid

        contour = kd.ImaginaryContour(β=β);
        grid = kd.ImaginaryTimeGrid(contour, nτ);

        # Hybridization propagator

        A_bethe = bethe_dos(t=t_bethe, ϵ=μ_bethe)
        Δ = kd.ImaginaryTimeGF(grid, 2) do t1, t2
            I(2) * V^2 * kd.dos2gf(A_bethe, β, t1.bpoint, t2.bpoint)
        end

        # Pseudo Particle Strong Coupling Expansion

        expansion = Expansion(H_imp, soi, grid, hybridization=Δ)
        ed = expansion.ed

        ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

        inchworm!(expansion, grid, orders, orders_bare, N_samples)

        normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(expansion.P), ed), ed)

        diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
        diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
        diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
        diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

        add_corr_operators!(expansion, (op.c(1), op.c_dag(1)))
        g = -correlator_2p(expansion, grid, orders_gf, N_samples)

        diff_g_nca = maximum(abs.(g_nca - g[1].mat.data[1, 1, :]))
        diff_g_oca = maximum(abs.(g_oca - g[1].mat.data[1, 1, :]))
        diff_g_tca = maximum(abs.(g_tca - g[1].mat.data[1, 1, :]))

        return real(diag(ρ_wrm)),
               diff_exa, diff_nca, diff_oca, diff_tca,
               diff_g_nca, diff_g_oca, diff_g_tca
    end

    @testset "order 1" begin
        nτ = 128
        orders = 0:1
        orders_gf = 0:0
        N_samples = 8 * 2^5
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca, diff_g_nca, diff_g_oca, diff_g_tca =
            run_bethe_gf(nτ, orders, orders, orders_gf, N_samples, μ_bethe)

        @test sum(ρ) ≈ 1
        @test ρ[2] ≈ ρ[3]

        @test diffs_nca < 2e-3
        @test diffs_nca < diffs_oca
        @test diffs_nca < diffs_tca
        @test diffs_nca < diffs_exa

        @test diff_g_nca < 3e-3
        @test diff_g_nca < diff_g_oca
    end

    @testset "order 2" begin
        nτ = 128
        orders = 0:2
        orders_gf = 0:1
        N_samples = 8 * 2^6
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca, diff_g_nca, diff_g_oca, diff_g_tca =
            run_bethe_gf(nτ, orders, orders, orders_gf, N_samples, μ_bethe)

        @test sum(ρ) ≈ 1
        @test ρ[2] ≈ ρ[3]

        @test diffs_oca < 2e-3
        @test diffs_oca < diffs_nca
        @test diffs_oca < diffs_tca
        @test diffs_oca < diffs_exa

        @test diff_g_oca < 1e-3
        @test diff_g_oca < diff_g_nca
    end

    @test_skip @testset "order 3" begin
        nτ = 128
        orders = 0:3
        orders_gf = 0:2
        N_samples = 8 * 2^6
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca, diff_g_nca, diff_g_oca, diff_g_tca =
            run_bethe_gf(nτ, orders, orders, orders_gf, N_samples, μ_bethe)

        @test sum(ρ) ≈ 1
        @test ρ[2] ≈ ρ[3]

        @test diffs_tca < 2e-3
        @test diffs_tca < diffs_nca
        @test diffs_tca < diffs_oca

        @test diff_g_tca < 7e-3
        @test diff_g_tca < diff_g_nca
        @test diff_g_tca < diff_g_oca
    end

end
