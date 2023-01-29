using MPI; MPI.Init()

import MD5
import HDF5; h5 = HDF5

import PyCall

using Test
using Printf

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
using  QInchworm.utility: inch_print

function semi_circular_g_tau(times, t, h, β)

    np = PyCall.pyimport("numpy")
    kernel = PyCall.pyimport("pydlr").kernel
    quad = PyCall.pyimport("scipy.integrate").quad

    #def eval_semi_circ_tau(tau, beta, h, t):
    #    I = lambda x : -2 / np.pi / t**2 * kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
    #    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    #    return g

    g_out = zero(times)
    
    for (i, tau) in enumerate(times)
        I = x -> -2 / np.pi / t^2 * kernel([tau/β], [β*x])[1, 1]
        g, res = quad(I, -t+h, t+h, weight="alg", wvar=(0.5, 0.5))
        g_out[i] = g
    end

    return g_out
end

function ρ_from_n_ref(ρ_wrm, n_ref)
    ρ_ref = zero(ρ_wrm)
    ρ_ref[1, 1] = (1 - n_ref) * (1 - n_ref)
    ρ_ref[2, 2] = n_ref * (1 - n_ref)
    ρ_ref[3, 3] = n_ref * (1 - n_ref)
    ρ_ref[4, 4] = n_ref * n_ref
    return ρ_ref
end

function ρ_from_ρ_ref(ρ_wrm, ρ_ref)
    ρ = zero(ρ_wrm)
    ρ[1, 1] = ρ_ref[1]
    ρ[2, 2] = ρ_ref[2]
    ρ[3, 3] = ρ_ref[3]
    ρ[4, 4] = ρ_ref[4]
    return ρ
end

function get_ρ_exact(ρ_wrm)
    n = 0.5460872495307262 # from DLR calc
    return ρ_from_n_ref(ρ_wrm, n)
end

function get_ρ_nca(ρ_wrm)
    rho_nca = [ 0.1961713995875524, 0.2474226001525296, 0.2474226001525296, 0.3089834001073883,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_nca)
end

function get_ρ_oca(ρ_wrm)
    rho_oca = [ 0.2018070389569783, 0.2476929924482211, 0.2476929924482211, 0.3028069761465793,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_oca)
end

function get_ρ_tca(ρ_wrm)
    rho_tca = [ 0.205163794520457, 0.2478638876741985, 0.2478638876741985, 0.2991084301311462,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_tca)
end

function run_hubbard_dimer(ntau, orders, N_chunks)

    β = 10.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0
    μ_bethe = 0.25

    # -- ED solution

    H_imp = -μ * (op.n(1) + op.n(2))
        
    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    
    soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed = KeldyshED.EDCore(H_imp, soi)
    
    # -- Hybridization propagator

    tau = [ real(im * τ.bpoint.val) for τ in grid ]
    delta_bethe = V^2 * semi_circular_g_tau(tau, t_bethe, μ_bethe, β)
    
    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> 1.0im * V^2 *
            semi_circular_g_tau(
                [-imag(t1.bpoint.val - t2.bpoint.val)],
                t_bethe, μ_bethe, β)[1],
        grid, 1, kd.fermionic, true)
    
    #println("=========================")
    #println(Δ)
    #println("=========================")
    #println(Δ_old)
    #exit()
    
    function reverse(g::kd.ImaginaryTimeGF)
        g_rev = deepcopy(g)
        τ_0, τ_β = first(g.grid), last(g.grid)
        for τ in g.grid
            g_rev[τ, τ_0] = g[τ_β, τ]
        end
        return g_rev
    end

    #println("=========================")
    #println(reverse(Δ))
    #println("=========================")

    # -- Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    ρ_0 = density_matrix(expansion.P0, ed)
    
    N_per_chunk = 8
    qmc_convergence_atol = 1e-15
    orders_bare = orders

    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_per_chunk,
                        N_chunks,
                        qmc_convergence_atol)

    ppgf.normalize!(expansion.P, β)
    ρ_wrm = density_matrix(expansion.P, ed)

    ρ_exa = get_ρ_exact(ρ_wrm)
    ρ_nca = get_ρ_nca(ρ_wrm)
    ρ_oca = get_ρ_oca(ρ_wrm)
    ρ_tca = get_ρ_tca(ρ_wrm)
    
    diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
    diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
    diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
    diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

    ρ_000 = real(LinearAlgebra.diag(ρ_0))
    ρ_exa = real(LinearAlgebra.diag(ρ_exa))
    ρ_nca = real(LinearAlgebra.diag(ρ_nca))
    ρ_oca = real(LinearAlgebra.diag(ρ_oca))
    ρ_tca = real(LinearAlgebra.diag(ρ_tca))
    ρ_wrm = real(LinearAlgebra.diag(ρ_wrm))
    
    if inch_print()
        @show ρ_000        
        @show ρ_nca
        @show ρ_oca
        @show ρ_tca
        @show ρ_exa
        @show ρ_wrm

        @show diff_nca
        @show diff_oca
        @show diff_tca
        @show diff_exa
    end

    return diff_exa, diff_nca, diff_oca, diff_tca
end

    
@testset "bethe_order1" begin
    
    ntau = 128
    orders = 0:1
    N_chunks = 2^6
    diffs_exa, diffs_nca, diffs_oca, diffs_tca = run_hubbard_dimer(ntau, orders, N_chunks)

    @test diffs_nca < 1e-3
    @test diffs_nca < diffs_oca 
    @test diffs_nca < diffs_tca
    @test diffs_nca < diffs_exa

end

@testset "bethe_order2" begin

    ntau = 128
    orders = 0:2
    N_chunks = 2^6
    diffs_exa, diffs_nca, diffs_oca, diffs_tca = run_hubbard_dimer(ntau, orders, N_chunks)

    @test diffs_oca < 1e-3
    @test diffs_oca < diffs_nca
    @test diffs_oca < diffs_tca
    @test diffs_oca < diffs_exa

end

@testset "bethe_order3" begin

    ntau = 128
    orders = 0:3
    N_chunks = 2^6
    diffs_exa, diffs_nca, diffs_oca, diffs_tca = run_hubbard_dimer(ntau, orders, N_chunks)

    @test diffs_tca < 2e-3
    @test diffs_tca < diffs_nca
    @test diffs_tca < diffs_oca
    @test diffs_tca < diffs_exa

end
