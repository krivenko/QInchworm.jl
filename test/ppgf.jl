using LinearAlgebra: Diagonal, ones, tr

using Keldysh; kd = Keldysh;
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: atomic_ppgf,
                      operator_product,
                      operator_matrix_representation,
                      total_density_operator,
                      first_order_spgf,
                      check_ppgf_real_time_symmetries

@testset "atomic ppgf" begin

    β = 10.

    U = +2.0 # Local interaction
    V = -0.1 # Hybridization
    B = +0.0 # Magnetic field
    μ = -0.1 # Chemical potential

    nt = 10
    ntau = 10

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

    # Check that atomic G0(β, 0) is proportinal to ρ
    function check_consistency_with_density_matrix(G0, ρ)
        for (G0_s, ρ_s) in zip(G0, ρ)
            t_0, t_beta = kd.branch_bounds(G0_s.grid, kd.imaginary_branch)
            @test ρ_s ≈ im * G0_s[t_beta, t_0]
        end
    end

    # Compute Tr[ρ c^+_1 c_2] using ED ρ and ppgf G0 cf spgf
    function check_consistency_n(G0, ed)
        grid = G0[1].grid
        t_0, t_beta = kd.branch_bounds(grid, kd.imaginary_branch)

        g_ref = ked.computegf(ed, grid, d, d);
        n_ref = im * g_ref[t_beta, t_0]

        idx1 = d
        idx2 = d

        n_rho::Complex = 0.
        n_G0::Complex = 0.

        for (sidx1, s) in enumerate(ed.subspaces)

            sidx2 = ked.c_connection(ed, idx1, sidx1)
            sidx2 === nothing && continue

            sidx3 = ked.cdag_connection(ed, idx2, sidx2)
            sidx3 != sidx1 && continue

            m_1 = ked.c_matrix(ed, idx1, sidx1)
            m_2 = ked.cdag_matrix(ed, idx2, sidx2)

            n_rho += tr(ρ[sidx1] * m_2 * m_1)
            n_G0 += tr(im * G0[sidx1][t_beta, t_0] * m_2 * m_1 )
        end

        @test n_rho ≈ n_ref
        @test n_G0 ≈ n_ref
    end

    # Check spgf from ED and 1st order Inch
    function check_consistency_first_order_spgf(G0, ed)
        grid = G0[1].grid
        for (o1, o2) in [(u, u), (u, d), (d, u), (d, d)]
            g = first_order_spgf(G0, ed, o1, o2);
            g_ref = ked.computegf(ed, grid, o1, o2);
            for z1 in grid, z2 in grid
                @test isapprox(g[z1, z2], g_ref[z1, z2], atol=1e-12, rtol=1-12)
            end
        end
    end

    @testset "Twisted Kadanoff-Baym-Keldysh contour" begin
        contour = kd.twist(kd.FullContour(tmax=30., β=β))
        grid = kd.FullTimeGrid(contour, nt, ntau)

        # Atomic propagator G0
        G0 = atomic_ppgf(grid, ed)
        @test check_ppgf_real_time_symmetries(G0, ed)

        check_consistency_with_density_matrix(G0, ρ)

        # Check that propagation from
        # - t on fwd branch over t_max
        # - to the same time t on bwd branch
        # is unity

        zb_max = grid[kd.backward_branch][1]
        zf_max = grid[kd.forward_branch][end]

        for (sidx, G_s) in enumerate(G0)
            for (zb, zf) in zip(reverse(grid[kd.backward_branch]), grid[kd.forward_branch])
                prod = im^2 * G_s[zb, zb_max] * G_s[zf_max, zf]
                I = Diagonal(ones(size(prod, 1)))
                @test prod ≈ I
            end
        end

        check_consistency_n(G0, ed)
        check_consistency_first_order_spgf(G0, ed)
    end

    @testset "Imaginary time" begin
        contour = kd.ImaginaryContour(β=β)
        grid = kd.ImaginaryTimeGrid(contour, ntau)

        # Atomic propagator G0
        G0 = atomic_ppgf(grid, ed)

        check_consistency_with_density_matrix(G0, ρ)
        check_consistency_n(G0, ed)
        check_consistency_first_order_spgf(G0, ed)
    end
end
