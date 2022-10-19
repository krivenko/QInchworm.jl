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

import QInchworm.KeldyshED_addons: reduced_density_matrix,
    density_matrix, reduced_ppgf, occupation_number_basis_ppgf

import QInchworm.spline_gf: SplineInterpolatedGF


import PyPlot as plt # DEBUG




function run_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol; interpolate_gfs=false)

    β = 1.0
    ϵ_1, ϵ_2 = 0.5, 2.0
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

    P0_dimer = ppgf.atomic_ppgf(grid, ed_dimer)
    P_red = reduced_ppgf(P0_dimer, ed_dimer, ed)
    
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

    if interpolate_gfs
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(reverse(Δ)))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=true)
    else
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])
    end

    if false
    # -- DEBUG

    τ = [ -imag(t.bpoint.val) for t in grid ]
    τ_0 = grid[1]

    for p0 in expansion.P0
        p_vec = cat([ p0[t, τ_0] for t in grid ]..., dims=3)
        i = 1
        plt.plot(τ, -imag(p_vec[i, i, 1:end]), "+", label="P_$i (0)")
    end

    grid_f = kd.ImaginaryTimeGrid(contour, ntau*100)

    τ_f = [ -imag(t.bpoint.val) for t in grid_f ]
    τ_f0 = grid_f[1]

    println("starting interpolation")
    if true
        for (s, p0) in enumerate(expansion.P0)
            p_vec = []
            for t in grid_f
                push!(p_vec, p0(t.bpoint, τ_f0.bpoint))
            end
            p_vec = [ val[1, 1] for val in p_vec ]
            plt.plot(τ_f, -imag(p_vec), "-", label="P_$s (spline)")

            p_vec = []
            for t in grid_f
                push!(p_vec, p0.GF(t.bpoint, τ_f0.bpoint))
            end
            p_vec = [ val[1, 1] for val in p_vec ]
            plt.plot(τ_f, -imag(p_vec), "-", label="P_$s (linear)")
        end
    end
    
    plt.legend(loc="best")
    plt.grid(true)
    plt.show()

    exit()
    end
    
    ρ_0 = density_matrix(expansion.P0, ed)
    
    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    if interpolate_gfs
        P = [ p.GF for p in expansion.P ]
        ppgf.normalize!(P, β) # DEBUG fixme!
        ρ_wrm = density_matrix(P, ed)
    else
        ppgf.normalize!(expansion.P, β) # DEBUG fixme!
        ρ_wrm = density_matrix(expansion.P, ed)
    end

    @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
    @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
    @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])
    
    diff = maximum(abs.(ρ_ref - ρ_wrm))
    @show diff
    
    P0 = occupation_number_basis_ppgf(expansion.P0, ed)
    if interpolate_gfs
        P = occupation_number_basis_ppgf(P, ed)
    else
        P = occupation_number_basis_ppgf(expansion.P, ed)
    end

    # -- Rip out initial derivative for P0 and P

    dP = ppgf.initial_ppgf_derivative(ed, β)    
    @show dP
    
    #exit()
    
    # -- DEBUG

    τ = [ -imag(t.bpoint.val) for t in grid ]
    τ_0 = grid[1]
    
    for dp in dP
        plt.plot(τ, -imag(-im .+ dp .* τ), "-", label="dP")
    end

    for i in 1:2
        p_vec = cat([ P0[t, τ_0] for t in grid ]..., dims=3)
        plt.plot(τ, -imag(p_vec[i, i, 1:end]), "+-", label="P_$i (0)")
    end

    for i in 1:2
        p_vec = cat([ P[t, τ_0] for t in grid ]..., dims=3)
        plt.plot(τ, -imag(p_vec[i, i, 1:end]), "x-", label="P_$i (inch)")
    end
    
    #for i in 1:2
    #    p_vec = cat([ P_red[t, τ_0] for t in grid ]..., dims=3)
    #    plt.plot(τ, -imag(p_vec[i, i, 1:end]), ".-", label="P_$i (red)")
    #end

    @show ρ_ref
    for i in 1:2
        plt.plot(τ[end], real(ρ_ref[i, i]), "o")
    end

    plt.legend(loc="best")
    plt.grid(true)
    plt.show()

    exit()
    
    return diff
end


@testset "inchworm_matsubara_dimer" begin    
    orders = 0:1
    orders_bare = 0:1
    qmc_convergence_atol = 1e-15

    ntau = 32
    #ntau = 128
    #ntau = 512
    N_per_chunk = 64
    N_chunks = 2
    #N_chunks = 128

    diff_interp = run_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks, qmc_convergence_atol, interpolate_gfs=true) 
    @test diff_interp < 1e-3
    @show diff_interp
    
    #diff_linear = run_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks, qmc_convergence_atol) 
    #@test diff_linear < 1e-3
    #@show diff_linear
end


function run_hubbard_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol)

    β = 1.0
    U = 4.0
    ϵ_1, ϵ_2 = 0.0 - 0.5*U, 0.0
    V_1 = 0.5
    V_2 = 0.5

    # -- ED solution

    H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))
    
    H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) + 
        V_1 * ( op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1) ) + 
        V_2 * ( op.c_dag(2) * op.c(4) + op.c_dag(4) * op.c(2) )
                             
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
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

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

    @printf "ρ_0   = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2]) real(ρ_0[3, 3]) real(ρ_0[4, 4])
    @printf "ρ_ref = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2]) real(ρ_ref[3, 3]) real(ρ_ref[4, 4])
    @printf "ρ_wrm = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2]) real(ρ_wrm[3, 3]) real(ρ_wrm[4, 4])
    
    diff = maximum(abs.(ρ_ref - ρ_wrm))
    @show diff

    return diff
end

@testset "inchworm_matsubara_hubbard_dimer" begin

    return
    
    qmc_convergence_atol = 1e-15

    ntau = 16
    N_per_chunk = 32
    N_chunks = 2

    orders_bare, orders = 0:1, 0:1
    diff_o1 = run_hubbard_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks, qmc_convergence_atol) 

    orders_bare, orders = 0:2, 0:2
    diff_o2 = run_hubbard_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks, qmc_convergence_atol) 
    
    @show diff_o1
    @show diff_o2

    @test diff_o1 < 3e-4
    @test diff_o2 < 3e-5
    @test diff_o1 > 8 * diff_o2
    
end

