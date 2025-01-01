# QInchworm.jl
#
# Copyright (C) 2021-2025 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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

using LinearAlgebra: Diagonal, ones, tr

using Keldysh; kd = Keldysh;
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: atomic_ppgf,
                      operator_product,
                      operator_matrix_representation,
                      total_density_operator,
                      first_order_spgf,
                      check_ppgf_real_time_symmetries

@testset "Atomic PPGF" begin

    β = 10.

    U = +2.0 # Local interaction
    V = -0.1 # Hybridization
    B = +0.0 # Magnetic field
    μ = -0.1 # Chemical potential

    tmax = 30.
    nt = 10
    nτ = 10

    # Hubbard-atom Hamiltonian

    H = U * (op.n("up") - 1/2) * (op.n("do") - 1/2)
    H += V * (op.c_dag("up") * op.c("do") + op.c_dag("do") * op.c("up"))
    H += B * (op.n("up") - op.n("do"))
    H += μ * (op.n("up") + op.n("do"))

    # Indices of fermionic states

    u = ked.Hilbert.IndicesType(["up"])
    d = ked.Hilbert.IndicesType(["do"])

    # Exact Diagonalization solver

    soi = ked.Hilbert.SetOfIndices([["up"], ["do"]]);
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

    # Check that atomic P0(β, 0) is proportional to ρ
    function check_consistency_with_density_matrix(P0, ρ)
        for (P0_s, ρ_s) in zip(P0, ρ)
            t_0, t_beta = kd.branch_bounds(P0_s.grid, kd.imaginary_branch)
            @test ρ_s ≈ im * P0_s[t_beta, t_0]
        end
    end

    # Compute Tr[ρ c^+_1 c_2] using ED ρ and PPGF P0 cf SPGF
    function check_consistency_n(P0, ed)
        grid = first(P0).grid
        t_0, t_beta = kd.branch_bounds(grid, kd.imaginary_branch)

        g_ref = ked.computegf(ed, grid, d, d);
        n_ref = im * g_ref[t_beta, t_0]

        idx1 = d
        idx2 = d

        n_rho::Complex = 0.
        n_P0::Complex = 0.

        for (sidx1, s) in enumerate(ed.subspaces)

            sidx2 = ked.c_connection(ed, idx1, sidx1)
            isnothing(sidx2) && continue

            sidx3 = ked.cdag_connection(ed, idx2, sidx2)
            sidx3 != sidx1 && continue

            m_1 = ked.c_matrix(ed, idx1, sidx1)
            m_2 = ked.cdag_matrix(ed, idx2, sidx2)

            n_rho += tr(ρ[sidx1] * m_2 * m_1)
            n_P0 += tr(im * P0[sidx1][t_beta, t_0] * m_2 * m_1 )
        end

        @test n_rho ≈ n_ref
        @test n_P0 ≈ n_ref
    end

    # Check SPGF from ED and 1st order Inch
    function check_consistency_first_order_spgf(P0, ed)
        grid = first(P0).grid
        for (o1, o2) in [(u, u), (u, d), (d, u), (d, d)]
            g = first_order_spgf(P0, ed, o1, o2);
            g_ref = ked.computegf(ed, grid, o1, o2);
            for z1 in grid, z2 in grid
                @test isapprox(g[z1, z2], g_ref[z1, z2], atol=1e-12, rtol=1-12)
            end
        end
    end

    @testset "Twisted Kadanoff-Baym-Keldysh contour" begin
        contour = kd.twist(kd.FullContour(tmax=tmax, β=β))
        grid = kd.FullTimeGrid(contour, nt, nτ)

        # Atomic propagator P0
        P0 = atomic_ppgf(grid, ed)
        @test check_ppgf_real_time_symmetries(P0, ed)

        check_consistency_with_density_matrix(P0, ρ)

        # Check that propagation from
        # - t on fwd branch over t_max
        # - to the same time t on bwd branch
        # is unity

        zb_max = grid[kd.backward_branch][1]
        zf_max = grid[kd.forward_branch][end]

        for (sidx, P_s) in enumerate(P0)
            for (zb, zf) in zip(reverse(grid[kd.backward_branch]), grid[kd.forward_branch])
                prod = im^2 * P_s[zb, zb_max] * P_s[zf_max, zf]
                I = Diagonal(ones(size(prod, 1)))
                @test prod ≈ I
            end
        end

        check_consistency_n(P0, ed)
        check_consistency_first_order_spgf(P0, ed)
    end

    @testset "Imaginary time" begin
        contour = kd.ImaginaryContour(β=β)
        grid = kd.ImaginaryTimeGrid(contour, nτ)

        # Atomic propagator P0
        P0 = atomic_ppgf(grid, ed)

        check_consistency_with_density_matrix(P0, ρ)
        check_consistency_n(P0, ed)
        check_consistency_first_order_spgf(P0, ed)
    end
end
