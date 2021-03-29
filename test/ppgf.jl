
import Test.@test
import Test.@testset

import LinearAlgebra: Diagonal, tr

import Keldysh; kd = Keldysh;

import KeldyshED; ked = KeldyshED;
import KeldyshED.Hilbert;
import KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf: atomic_ppgf
import QInchworm.ppgf: operator_product
import QInchworm.ppgf: operator_matrix_representation
import QInchworm.ppgf: total_density_operator
import QInchworm.ppgf: first_order_spgf

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
    
    g = KeldyshED.computegf(ed, grid, [(d, d)], β)[1];
    
    # Atomic propagator G0, array of (matrix valued) TimeGF one for each ed subspace
    
    G0 = [ kd.TimeGF(grid, length(s)) for s in ed.subspaces ];
    
    # Compute atomic propagator for all times
    
    Z = KeldyshED.partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[G0(β)]=Tr[ρ]=1)

    t_β = grid[kd.imaginary_branch][end]
    N = operator_matrix_representation(total_density_operator(ed), ed)
    
    for (G0_s, E, n) in zip(G0, KeldyshED.energies(ed), N)
        ξ = (-1)^n[1,1] # Statistics sign        
        for t1 in grid, t2 in grid[1:t1.idx]
            Δt = t1.val.val - t2.val.val
            if t1.val.domain == kd.forward_branch && 
               t2.val.domain != kd.forward_branch                
                Δt += -im*β
            end            
            sign = ξ^(t1.idx > t_β.idx && t_β.idx >= t2.idx)
            G0_s[t1, t2] = -im * sign * Diagonal(exp.(-1im * Δt * (E .+ λ)))
        end
    end
    
    t_0, t_beta = kd.branch_bounds(grid, kd.imaginary_branch)
    
    for (G0_s, ρ_s) in zip(G0, ρ)
        @test ρ_s ≈ im * G0_s[t_beta, t_0]
    end

    G0_ref = atomic_ppgf(grid, ed, β)
    
    for (G0_s, G0_s_ref) in zip(G0, G0_ref)
        @test G0_s ≈ G0_s_ref
    end

    # -- Compute Tr[\rho c^+_1 c_2]

    n_ref = kd.density(g)[1]
    
    idx1 = d
    idx2 = d

    n::Complex = 0.

    for (sidx1, s) in enumerate(ed.subspaces)

        sidx2 = ked.c_connection(ed, idx1, sidx1)
        sidx2 == nothing && continue

        sidx3 = ked.cdag_connection(ed, idx2, sidx2)
        sidx3 != sidx1 && continue

        m_1 = ked.c_matrix(ed, idx1, sidx1)
        m_2 = ked.cdag_matrix(ed, idx2, sidx2)

        n += tr(ρ[sidx1] * m_2 * m_1)
    end

    @test n ≈ n_ref

    # -- Compute single-particle Green's function using G0

    g_tau = Vector{Real}()

    for tau_i in getindex(grid, kd.imaginary_branch)
        g_tau_i = 0.0im
        
        for (sidx1, s) in enumerate(ed.subspaces)

            sidx2 = ked.cdag_connection(ed, idx1, sidx1)
            sidx2 == nothing && continue

            sidx3 = ked.c_connection(ed, idx2, sidx2)
            sidx3 != sidx1 && continue

            m_1 = ked.cdag_matrix(ed, idx1, sidx1)
            m_2 = ked.c_matrix(ed, idx2, sidx2)

            prod = (im)^2 * G0[sidx1][t_beta, tau_i] * m_2 * G0[sidx2][tau_i, t_0] * m_1
            
            # -- This is the general configuration evaluator function
            prod_ref = operator_product(
                ed, G0, sidx1, t_0, t_beta, [(t_0, +1, idx1),(tau_i, -1, idx2)])
            
            @test isapprox(prod, prod_ref, atol=1e-12, rtol=1e-12)

            g_tau_i -= tr(prod)
        end    
        push!(g_tau, g_tau_i)
    end
    
    @test isapprox(g_tau, g[:matsubara], atol=1e-12, rtol=1-12)

    g_ref = first_order_spgf(G0, ed, d, d)

    @test isapprox(g, g_ref, atol=1e-12, rtol=1-12)
    
end
