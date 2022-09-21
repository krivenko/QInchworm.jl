using Test
using Printf

import LinearAlgebra; trace = LinearAlgebra.tr
import LinearAlgebra; diag = LinearAlgebra.diag
import LinearAlgebra; diagm = LinearAlgebra.diagm

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.configuration: Expansion, InteractionPair

import QInchworm.topology_eval: get_topologies_at_order,
                                get_diagrams_at_order

import QInchworm.inchworm: InchwormOrderData,
                           inchworm_step,
                           inchworm_step_bare,
                           inchworm_matsubara!

import QInchworm.ppgf

import QInchworm.KeldyshED_addons: reduced_density_matrix, density_matrix

function run_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol)

    β = 1.0
    ϵ_1, ϵ_2 = 0.0, 2.0
    V = 0.5

    # -- ED solution

    H_dimer = ϵ_1 * op.n(1) + ϵ_2 * op.n(2) + V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)
    
    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    
    H = ϵ_1 * op.n(1)
    soi = KeldyshED.Hilbert.SetOfIndices([[1]])
    ed = KeldyshED.EDCore(H, soi)

    ρ_ref = Array{ComplexF64}( reduced_density_matrix(ed_dimer, ed, β) )
    
    # -- Hybridization propagator
    
    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ_2, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ_2),
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

    ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    ρ_0 = density_matrix(expansion.P0, ed, β)

    #@show eigenstate_density_matrix(expansion.P0)
    #@show eigenstate_density_matrix(expansion.P)
    
    #@test ρ_ref ≈ ρ_0
    
    #exit()
    
    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    #@show eigenstate_density_matrix(expansion.P)

    #ρ_wrm_raw = density_matrix(expansion.P, ed, β)
    ppgf.normalize!(expansion.P, β)
    ρ_wrm = density_matrix(expansion.P, ed, β)

    # ρ_ref = 0.4928078037213840 0.5071921962786160 
    #[array([[0.49284566]]), array([[0.50715434]])] cthyb
    
    #a = 0.4928077555264891
    #b = 0.5071922444735112
    #ρ_nca = [[a, 0.], [0., b]]

    #@show ρ_0
    #@show ρ_ref
    #@show ρ_nca
    #@show ρ_wrm

    @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
    @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
    #@printf "ρ_nca = %16.16f %16.16f \n" ρ_nca[1][1] ρ_nca[2][2]    
    @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])
    #@printf "ρ_wrm = %16.16f %16.16f (un-normalized)\n" real(ρ_wrm_raw[1, 1]) real(ρ_wrm_raw[2, 2])
    
    #diff_ref = maximum(abs.(ρ_ref - ρ_0))
    diff = maximum(abs.(ρ_ref - ρ_wrm))
    #@show diff_ref
    @show diff
    return diff
end


@testset "inchworm_matsubara_dimer" begin

    return
    
    orders = 0:1
    orders_bare = 0:1
    qmc_convergence_atol = 1e-15

    ntau = 32
    N_per_chunk = 64
    N_chunks = 2

    diff = run_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks, qmc_convergence_atol) 
    @test diff < 1e-3
    
end


@testset "inchworm_matsubara_plot" begin

    return
    
    orders = 0:1
    orders_bare = 0:1
    qmc_convergence_atol = 1e-15

    #ntaus = [16, 32, 64, 128, 256, 512, 1024]
    #ntaus = [16, 32, 64, 128, 256]
    ntaus = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #N_chunkss = [0, 1, 2, 4, 8, 16]
    N_chunkss = [0, 1, 2, 4, 8, 16] 
    #N_chunkss = [1, 2, 4, 8]
    #max_chunkss = [1]

    @show ntaus
    @show N_chunkss

    import PyPlot as plt

    #for ntau in ntaus
    for N_chunks in N_chunkss

        N_per_chunk = 8
        
        #diffs = [ run_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol) for max_chunks in max_chunkss ]
        diffs = [ run_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks, qmc_convergence_atol) for ntau in ntaus ]

        #@show ntau
        #@show max_chunkss

        @show N_chunks
        @show ntaus
        @show diffs

        #N = max_chunkss .* ntau .* N_chunk
        N = N_chunks .* ntaus .* N_per_chunk
        
        #plt.loglog(N, diffs, "-o", label="n_tau = $ntau")
        plt.loglog(ntaus, diffs, "-o", label="N_chunks = $N_chunks")
        
    end

    plt.legend()
    plt.xlabel("N_tau")
    plt.ylabel("Err")
    #plt.ylim(bottom=0)
    #plt.xlim(left=0)
    plt.axis("image")
    plt.grid(true)
    plt.show()
    
end

function run_hubbard_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol)

    β = 1.0
    U = 0.0
    ϵ_1, ϵ_2 = 0.0, 2.0
    V_1 = 0.0
    V_2 = 0.5

    # -- ED solution

    H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))
    
    H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) + 
        V_1 * ( op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1) ) + 
        V_2 * ( op.c_dag(2) * op.c(4) + op.c_dag(2) * op.c(4) )
                             
    soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2], [3], [4]])
    ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)
    
    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    
    soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed = KeldyshED.EDCore(H_imp, soi)

    ρ_ref = Array{ComplexF64}( reduced_density_matrix(ed_dimer, ed, β) )
    
    # -- Hybridization propagator
    
    Δ_1 = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V_1^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ_2, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ_2),
        grid, 1, kd.fermionic, true)

    Δ_2 = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V_2^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ_2, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ_2),
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

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ_1)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ_1))

    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ_2)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ_2))

    #expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])
    #expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd])
    expansion = Expansion(ed, grid, [ip_2_fwd, ip_2_bwd])

    ρ_0 = density_matrix(expansion.P0, ed, β)

    #@show eigenstate_density_matrix(expansion.P0)
    #@show eigenstate_density_matrix(expansion.P)
    
    #@test ρ_ref ≈ ρ_0
    
    #exit()
    
    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    ppgf.normalize!(expansion.P, β)
    ρ_wrm = density_matrix(expansion.P, ed, β)

    # ρ_ref = 
    #[array([[0.24646073]]), array([[0.24646073]]), array([[0.25353927]]), array([[0.25353927]])]
    #[array([[0.24641]]), array([[0.24641]]), array([[0.25359]]), array([[0.25359]])]
    #[array([[0.24641498]]), array([[0.25358502]]), array([[0.24641498]]), array([[0.25358502]])]

    @show ρ_0
    @show ρ_ref
    @show ρ_wrm

    @printf "ρ_0   = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2]) real(ρ_0[3, 3]) real(ρ_0[4, 4])
    @printf "ρ_ref = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2]) real(ρ_ref[3, 3]) real(ρ_ref[4, 4])
    @printf "ρ_wrm = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2]) real(ρ_wrm[3, 3]) real(ρ_wrm[4, 4])
    
    diff = maximum(abs.(ρ_ref - ρ_wrm))
    @show diff

    return diff
end

@testset "inchworm_matsubara_dimer" begin

    return
    
    orders = 0:1
    orders_bare = 0:1
    qmc_convergence_atol = 1e-15

    ntau = 128
    N_per_chunk = 128
    N_chunks = 2

    diff = run_hubbard_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks, qmc_convergence_atol) 
    @test diff < 1e-3
    
end


function get_reduced_density_matrix_hubbard_dimer(β, U, ϵ_1, ϵ_2, V_1, V_2)

    H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))
    
    H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) + 
        V_1 * ( op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1) ) + 
        V_2 * ( op.c_dag(2) * op.c(4) + op.c_dag(2) * op.c(4) )
                             
    soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2], [3], [4]])
    ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)
    
    soi_small = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed_small = KeldyshED.EDCore(H_imp, soi_small)

    ρ_reduced = reduced_density_matrix(ed_dimer, ed_small, β)

    return ρ_reduced
end


@testset "reduced density matrix (hybridized hubbard dimer)" begin

    β = 1.0
    U = 0.0
    ϵ_1, ϵ_2 = 0.0, 2.0

    V_1, V_2 = 0.5, 0.0
    ρ_1 = get_reduced_density_matrix_hubbard_dimer(β, U, ϵ_1, ϵ_2, V_1, V_2)

    V_1, V_2 = 0.0, 0.5
    ρ_2 = get_reduced_density_matrix_hubbard_dimer(β, U, ϵ_1, ϵ_2, V_1, V_2)

    # Permutation matrix for switching |1> and |2>
    P = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    #@show P

    ρ_1 = P * ρ_1 * P # Permute states in ρ_1 so that it becomes identical to ρ_2

    @test all(ρ_1 - diagm(diag(ρ_1)) .≈ 0.)
    @test all(ρ_2 - diagm(diag(ρ_2)) .≈ 0.)
    
    @show diag(ρ_1)
    @show diag(ρ_2)

    @test trace(ρ_2) ≈ 1.0
    @test trace(ρ_1) ≈ 1.0

    @test ρ_1 ≈ ρ_2
        
end
