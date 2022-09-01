using Test
using Printf

import LinearAlgebra; trace = LinearAlgebra.tr

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

    #@printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
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


@testset "inchworm_matsubara" begin

    orders = 0:1
    orders_bare = 0:1
    #N_chunk = 10000
    N_chunk = 10 * 2
    max_chunks = 5
    qmc_convergence_atol = 1e-15

    ntaus = 2 .^ (2:15)
    #ntaus = [16 * 2 * 2 * 2 * 2 * 2]
    @show ntaus
    #exit()

    diffs = [ run_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol) for ntau in ntaus ]

    @show diffs
end
