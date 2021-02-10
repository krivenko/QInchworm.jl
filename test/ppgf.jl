
import Test.@test
import Test.@testset

import LinearAlgebra: Diagonal

import Keldysh; kd = Keldysh;

import KeldyshED;
import KeldyshED.Hilbert;
import KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf: AtomicPseudoParticleGF

@testset "atomic ppgf" begin

    β = 10.

    U = +2.0 # Local interaction
    V = -0.1 # Hybridization
    B = +0.0 # Magnetic field

    # Hubbard-atom Hamiltonian

    H = U * (op.n("up") - 1/2) * (op.n("do") - 1/2) 
    H += V * (op.c_dag("up") * op.c("do") + op.c_dag("do") * op.c("up")) 
    H += B * (op.n("up") - op.n("do"))

    # Exact Diagonalization solver
    
    soi = KeldyshED.Hilbert.SetOfIndices([["up"], ["do"]]);
    ed = KeldyshED.EDCore(H, soi)
    
    ρ = KeldyshED.density_matrix(ed, β)
    
    # Real-time Kadanoff-Baym contour
    
    contour = kd.twist(kd.Contour(kd.full_contour, tmax=30., β=β));
    grid = kd.TimeGrid(contour, npts_real=100, npts_imag=100);
    
    # Single particle Green's function
    
    u = KeldyshED.Hilbert.IndicesType(["up"])
    d = KeldyshED.Hilbert.IndicesType(["do"])
    
    g = KeldyshED.computegf(ed, grid, [(d, d)], β)[1];
    
    # Atomic propagator G0, array of (matrix valued) TimeGF one for each ed subspace
    
    G0 = [ kd.TimeGF(grid, length(s)) for s in ed.subspaces ];
    
    # Compute atomic propagator for all times
    
    Z = KeldyshED.partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[G0(β)]=Tr[ρ]=1)
    
    for (G0_s, E) in zip(G0, KeldyshED.energies(ed))
        for t1 in grid, t2 in grid
            Δt = t1.val.val - t2.val.val
            G0_s[t1, t2] = Diagonal(exp.(-1im * Δt * (E .+ λ)))
        end
    end
    
    t_0, t_beta = kd.branch_bounds(grid, kd.imaginary_branch)
    
    for (G0_s, ρ_s) in zip(G0, ρ)
        @test ρ_s ≈ G0_s[t_beta, t_0]
    end

    G0_ref = AtomicPseudoParticleGF(grid, ed, β)
    for (G0_s, G0_s_ref) in zip(G0, G0_ref)
        @test G0_s ≈ G0_s_ref
    end

end
