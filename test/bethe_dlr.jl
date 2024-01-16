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
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.inchworm: inchworm!

#using QInchworm.utility: ph_conj

import Lehmann; le = Lehmann
using QInchworm.keldysh_dlr: DLRImaginaryTimeGrid, DLRImaginaryTimeGF, ph_conj

# Solve non-interacting two fermion impurity model coupled to semi-circular (Bethe lattice)
# hybridization functions.
#
# Performing two kinds of tests:
#
# 1. Checking that the Inchworm expansion does not break particle-hole symmetry for an
# impurity model with PH-symmetry.
#
# 2. Compare to numerically exact results for the 1st, 2nd and 3rd order dressed
# self-consistent expansion for the many-body density matrix (computed using DLR elsewhere).
#
# Note that the 1, 2, 3 order density matrix differs from the exact density matrix of
# the non-interacting system, since the low order expansions introduce "artificial"
# effective interactions between hybridization insertions.
@testset "Bethe" verbose=true begin

    # Reference results from DLR calculations
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
        t_bethe = 0.5

        # Hamiltonian

        H_imp = -μ * (op.n(1) + op.n(2))
        soi = ked.Hilbert.SetOfIndices([[1], [2]])
        ed = ked.EDCore(H_imp, soi)
        
        # Imaginary time grid

        contour = kd.ImaginaryContour(β=β)
        grid = kd.ImaginaryTimeGrid(contour, nτ)

        dlr = le.DLRGrid(Euv=1.25, β=β, isFermi=true, rtol=1e-6, rebuild=true, verbose=false)
        dlr_grid = DLRImaginaryTimeGrid(contour, dlr)

        # Hybridization propagator

        A_bethe = bethe_dos(t=t_bethe, ϵ=μ_bethe)
        Δ = DLRImaginaryTimeGF(dlr_grid, 1, kd.fermionic, true) do t1, t2
        #Δ = ImaginaryTimeGF(grid, 1, kd.fermionic, true) do t1, t2
            V^2 * kd.dos2gf(A_bethe, β, t1.bpoint, t2.bpoint)
        end

        # Pseudo Particle Strong Coupling Expansion

        ip_fwd_1 = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_bwd_1 = InteractionPair(op.c(1), op.c_dag(1), ph_conj(Δ))

        ip_fwd_2 = InteractionPair(op.c_dag(2), op.c(2), Δ)
        ip_bwd_2= InteractionPair(op.c(2), op.c_dag(2), ph_conj(Δ))

        expansion = Expansion(ed, grid, [ip_fwd_1, ip_bwd_1, ip_fwd_2, ip_bwd_2])
        
        ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

        inchworm!(expansion, grid, orders, orders_bare, N_samples)

        normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(expansion.P), ed), ed)

        diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
        diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
        diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
        diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

        return real(diag(ρ_wrm)), diff_exa, diff_nca, diff_oca, diff_tca
    end

    @testset "PH symmetry" begin
        nτ = 3
        N_samples = 2^4
        μ_bethe = 0.0

        tests = [
            (0:0, 0:0),
            (0:1, 0:0),
            (0:0, 0:1),
            (0:2, 0:0),
            (0:0, 0:2),
            (0:3, 0:0),
            (0:0, 0:3),
            (0:4, 0:0),
            (0:0, 0:4)
            ]

        for (orders_bare, orders) in tests
            ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
                run_bethe(nτ, orders, orders_bare, N_samples, μ_bethe)
            @test ρ ≈ [0.25, 0.25, 0.25, 0.25]
        end
    end

    @testset "order 1" begin
        nτ = 128
        orders = 0:1
        N_samples = 8 * 2^5
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
            run_bethe(nτ, orders, orders, N_samples, μ_bethe)

        @test sum(ρ) ≈ 1
        @test ρ[2] ≈ ρ[3]
        @test diffs_nca < 2e-3
        @test diffs_nca < diffs_oca
        @test diffs_nca < diffs_tca
        @test diffs_nca < diffs_exa
    end

    @testset "order 2" begin
        nτ = 128
        orders = 0:2
        N_samples = 8 * 2^5
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
            run_bethe(nτ, orders, orders, N_samples, μ_bethe)

        @test sum(ρ) ≈ 1
        @test ρ[2] ≈ ρ[3]
        @test diffs_oca < 4e-3
        @test diffs_oca < diffs_nca
        @test diffs_oca < diffs_tca
        @test diffs_oca < diffs_exa
    end

    @testset "order 3" begin
        nτ = 128
        orders = 0:3
        N_samples = 8 * 2^5
        μ_bethe = 0.25

        ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca =
            run_bethe(nτ, orders, orders, N_samples, μ_bethe)

        @test sum(ρ) ≈ 1
        @test ρ[2] ≈ ρ[3]
        @test diffs_tca < 4e-3
        @test diffs_tca < diffs_nca
        @test diffs_tca < diffs_oca
        #@test diffs_tca < diffs_exa
    end

end
