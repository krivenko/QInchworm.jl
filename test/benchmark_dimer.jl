using Test
using Printf
import PyPlot as plt

import LinearAlgebra; trace = LinearAlgebra.tr

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf
import QInchworm.configuration: Expansion, InteractionPair
import QInchworm.topology_eval: get_topologies_at_order,
                                get_diagrams_at_order
import QInchworm.inchworm: InchwormOrderData,
                           inchworm_step,
                           inchworm_step_bare,
                           inchworm_matsubara!
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
    
    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    ppgf.normalize!(expansion.P, β)
    ρ_wrm = density_matrix(expansion.P, ed, β)

    @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
    @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
    @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])
    
    diff = maximum(abs.(ρ_ref - ρ_wrm))
    @show diff
    return diff
end


@testset "inchworm_matsubara_plot" begin
    
    orders = 0:1
    orders_bare = 0:1
    qmc_convergence_atol = 1e-15

    #ntaus = [16, 32, 64, 128, 256, 512, 1024]
    #ntaus = [16, 32, 64, 128, 256]
    #ntaus = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    ntaus = [4, 8, 16, 32, 64, 128]
    #N_chunkss = [0, 1, 2, 4, 8, 16]
    N_chunkss = [0, 1, 2, 4, 8, 16] 
    #N_chunkss = [1, 2, 4, 8]
    #max_chunkss = [1]

    @show ntaus
    @show N_chunkss

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
