using MPI

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


function run_hubbard_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol)

    β = 10.0
    #U = 0.0
    #ϵ_1, ϵ_2 = 0.0 - 0.5*U, 2.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0
    μ_bethe = 0.25

    # -- ED solution

    #H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))
    H_imp = -μ * (op.n(1) + op.n(2))
    
    #H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) + 
    #    V_1 * ( op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1) ) + 
    #    V_2 * ( op.c_dag(2) * op.c(4) + op.c_dag(4) * op.c(2) )
                             
    #soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2], [3], [4]])
    #ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)
    
    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    
    soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed = KeldyshED.EDCore(H_imp, soi)

    #ρ_ref = Array{ComplexF64}( reduced_density_matrix(ed_dimer, ed, β) )
    
    # -- Hybridization propagator

    tau = [ real(im * τ.bpoint.val) for τ in grid ]
    #println(tau)
    delta_bethe = V^2 * semi_circular_g_tau(tau, t_bethe, μ_bethe, β)
    #println(delta_bethe)
    #exit()
    
    #Δ_old = kd.ImaginaryTimeGF(
    #    (t1, t2) -> -1.0im * V^2 *
    #        (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(μ_bethe, contour.β)) *
    #        exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * μ_bethe),
    #    grid, 1, kd.fermionic, true)

    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> 1.0im * V^2 *
            semi_circular_g_tau(
                [-imag(t1.bpoint.val - t2.bpoint.val)],
                t_bethe, μ_bethe, β)[1],
        grid, 1, kd.fermionic, true)
    
    #println(Δ)
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
    
    # -- Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    ρ_0 = density_matrix(expansion.P0, ed)
    
    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    ppgf.normalize!(expansion.P, β)
    ρ_wrm = density_matrix(expansion.P, ed)

    ρ_ref = zero(ρ_wrm)
    n_ref = 0.5460872495307262
    ρ_ref[1, 1] = (1 - n_ref) * (1 - n_ref)
    ρ_ref[2, 2] = n_ref * (1 - n_ref)
    ρ_ref[3, 3] = n_ref * (1 - n_ref)
    ρ_ref[4, 4] = n_ref * n_ref
    
    diff = maximum(abs.(ρ_ref - ρ_wrm))

    if inch_print()
        @printf "ρ_0   = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2]) real(ρ_0[3, 3]) real(ρ_0[4, 4])
        @printf "ρ_ref = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2]) real(ρ_ref[3, 3]) real(ρ_ref[4, 4])
        @printf "ρ_wrm = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2]) real(ρ_wrm[3, 3]) real(ρ_wrm[4, 4])
        @show diff
    end

    return diff
end

function run_ntau_calc(ntau::Integer, orders, N_chunkss)

    comm_root = 0
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    orders_bare = orders
    qmc_convergence_atol = 1e-15
    N_per_chunk = 8

    # -- Do calculation here

    diffs = [ run_hubbard_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks,
                                qmc_convergence_atol) for N_chunks in N_chunkss ]

    diff_0 = run_hubbard_dimer(ntau, orders, orders_bare, N_per_chunk, 0, qmc_convergence_atol)
    
    if comm_rank == comm_root
        @show diffs

        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        max_order = maximum(orders)
        filename = "data_FH_dimer_ntau_$(ntau)_maxorder_$(max_order)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")

        g = h5.create_group(fid, "data")

        h5.attributes(g)["qmc_convergence_atol"] = qmc_convergence_atol
        h5.attributes(g)["ntau"] = ntau
        h5.attributes(g)["N_per_chunk"] = N_per_chunk
        h5.attributes(g)["diff_0"] = diff_0

        g["orders"] = collect(orders)
        g["orders_bare"] = collect(orders_bare)
        g["N_chunkss"] = N_chunkss

        g["diffs"] = diffs
        
        h5.close(fid)
        
    end        
end


MPI.Init()

#ntaus = 2 .^ range(4, 12)
ntaus = 2 .^ range(3, 6)
#ntaus = [32]
#ntaus = [128]

N_chunkss = 2 .^ range(0, 7)
#N_chunkss = 2 .^ range(8, 10)
#N_chunkss = 2 .^ range(1, 10)
#N_chunkss = [2^5]

#orderss = [0:1, 0:2, 0:3]
#orderss = [0:1, 0:2, 0:3]
orderss = [0:4]

if inch_print()
    @show ntaus
    @show N_chunkss
    @show orderss
end

#exit()

for ntau in ntaus
    for orders in orderss
        run_ntau_calc(ntau, orders, N_chunkss)
    end
end
