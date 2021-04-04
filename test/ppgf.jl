
import Test.@test
import Test.@testset

import LinearAlgebra: Diagonal, ones, tr, conj

import Keldysh; kd = Keldysh;

import KeldyshED; ked = KeldyshED;
import KeldyshED.Hilbert;
import KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf: atomic_ppgf
import QInchworm.ppgf: operator_product
import QInchworm.ppgf: operator_matrix_representation
import QInchworm.ppgf: total_density_operator
import QInchworm.ppgf: first_order_spgf


function check_ppgf_real_time_symmetries(G, ed)

    grid = G[1].grid
    
    grid_bwd = grid[kd.backward_branch]
    zb_i, zb_f = grid_bwd[1], grid_bwd[end]

    grid_fwd = grid[kd.forward_branch]
    zf_i, zf_f = grid_fwd[1], grid_fwd[end]

    # Symmetry between G_{--} and G_{++}

    for zb_1 in grid_bwd
        for zb_2 in grid_bwd[1:zb_1.idx]
            @test zb_1.idx >= zb_2.idx

            zf_1 = grid[zf_f.idx - zb_1.idx + 1]
            zf_2 = grid[zf_f.idx - zb_2.idx + 1]
        
            @test zb_1.val.val ≈ zf_1.val.val
            @test zb_2.val.val ≈ zf_2.val.val
        
            @test zf_2.idx >= zf_1.idx
        
            for g_s in G
                @test g_s[zb_1, zb_2] ≈ -conj(g_s[zf_2, zf_1])
            end
        end
    end

    # Symmetry along anti-diagonal of G_{+-}

    for zf_1 in grid_fwd
        for zb_1 in grid_bwd[1:zf_f.idx - zf_1.idx + 1]
        
            zf_2 = grid[zf_f.idx - (zb_1.idx - zb_i.idx)]
            zb_2 = grid[zb_f.idx - (zf_1.idx - zf_i.idx)]
        
            @test zf_1.val.val ≈ zb_2.val.val
            @test zb_1.val.val ≈ zf_2.val.val
        
            for g_s in G
                @test g_s[zf_1, zb_1] ≈ -conj(g_s[zf_2, zb_2])
            end
        end
    end

    # Symmetry between G_{M-} and G_{+M}

    z_0 = grid[kd.imaginary_branch][1]
    z_β = grid[kd.imaginary_branch][end]
    β = im * z_β.val.val

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for zf in grid_bwd
    
        zb = grid[zf_f.idx - zf.idx + 1]
        @test zf.val.val ≈ zb.val.val
    
        for τ in grid[kd.imaginary_branch]
            τ_β = grid[z_β.idx - (τ.idx - z_0.idx)]
            @test τ_β.val.val ≈ -im*β - τ.val.val 
        
            for (sidx, g_s) in enumerate(G)
                ξ = (-1)^N[sidx][1, 1]
                @test g_s[τ, zf] ≈ -ξ * conj(g_s[zb, τ_β])
            end
        end
    end
    return true
end

@testset "atomic ppgf" begin

    β = 10.

    U = +2.0 # Local interaction
    V = -0.1 # Hybridization
    B = +0.0 # Magnetic field
    μ = -0.1 # Chemical potential

    # Hubbard-atom Hamiltonian

    H = U * (op.n("up") - 1/2) * (op.n("do") - 1/2) 
    H += V * (op.c_dag("up") * op.c("do") + op.c_dag("do") * op.c("up")) 
    H += B * (op.n("up") - op.n("do"))
    H += μ * (op.n("up") + op.n("do"))

    # Exact Diagonalization solver
    
    soi = KeldyshED.Hilbert.SetOfIndices([["up"], ["do"]]);
    ed = KeldyshED.EDCore(H, soi)
    ρ = KeldyshED.density_matrix(ed, β)
    
    # Real-time Kadanoff-Baym contour
    
    contour = kd.twist(kd.Contour(kd.full_contour, tmax=30., β=β));
    grid = kd.TimeGrid(contour, npts_real=10, npts_imag=10);
    
    # Single particle Green's function
    
    u = KeldyshED.Hilbert.IndicesType(["up"])
    d = KeldyshED.Hilbert.IndicesType(["do"])
        
    # Atomic propagator G0, array of (matrix valued) TimeGF one for each ed subspace

    G0 = atomic_ppgf(grid, ed, β)
    @test check_ppgf_real_time_symmetries(G0, ed)
    
    t_0, t_beta = kd.branch_bounds(grid, kd.imaginary_branch)
    
    for (G0_s, ρ_s) in zip(G0, ρ)
        @test ρ_s ≈ im * G0_s[t_beta, t_0]
    end

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

    # -- Compute Tr[\rho c^+_1 c_2] using ED ρ and ppgf G0 cf spgf

    g_ref = KeldyshED.computegf(ed, grid, [(d, d)], β)[1];
    n_ref = kd.density(g_ref)[1]
    
    idx1 = d
    idx2 = d

    n_rho::Complex = 0.
    n_G0::Complex = 0.

    for (sidx1, s) in enumerate(ed.subspaces)

        sidx2 = ked.c_connection(ed, idx1, sidx1)
        sidx2 == nothing && continue

        sidx3 = ked.cdag_connection(ed, idx2, sidx2)
        sidx3 != sidx1 && continue

        m_1 = ked.c_matrix(ed, idx1, sidx1)
        m_2 = ked.cdag_matrix(ed, idx2, sidx2)

        n_rho += tr(ρ[sidx1] * m_2 * m_1)
        n_G0 += tr(im * G0[sidx1][t_beta, t_0] * m_2 * m_1 )
    end

    @test n_rho ≈ n_ref
    @test n_G0 ≈ n_ref

    # Check spgf from ED and 1st order Inch
    
    for (o1, o2) in [(u, u), (u, d), (d, u), (d, d)]
        g = first_order_spgf(G0, ed, o1, o2);
        g_ref = ked.computegf(ed, grid, [(o1, o2)], β)[1];
        @test isapprox(g, g_ref, atol=1e-12, rtol=1-12)
    end
    
end
