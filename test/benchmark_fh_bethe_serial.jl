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

function run_ntau_calc(ntau::Integer, orders, N_chunkss)

    comm_root = 0
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    orders_bare = orders
    qmc_convergence_atol = 1e-15
    N_per_chunk = 8

    # -- Do calculation here
    diffs_exa = Array{Float64}(undef, length(N_chunkss))
    diffs_nca = Array{Float64}(undef, length(N_chunkss))
    diffs_oca = Array{Float64}(undef, length(N_chunkss))
    diffs_tca = Array{Float64}(undef, length(N_chunkss))

    for (idx, N_chunks) in enumerate(N_chunkss)
        diffs_exa[idx], diffs_nca[idx], diffs_oca[idx], diffs_tca[idx] =
            run_hubbard_dimer(ntau, orders, orders_bare,
                              N_per_chunk, N_chunks, qmc_convergence_atol)
    end
        
    diff_0_exa, diff_0_nca, diff_0_oca, diff_0_tca =
        run_hubbard_dimer(ntau, orders, orders_bare,
                          N_per_chunk, 0, qmc_convergence_atol)

    #diffs = [ run_hubbard_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks,
    #                            qmc_convergence_atol) for N_chunks in N_chunkss ]

    #diff_0 = run_hubbard_dimer(ntau, orders, orders_bare, N_per_chunk, 0, qmc_convergence_atol)
    
    
    if comm_rank == comm_root

        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs_exa)))
        max_order = maximum(orders)
        filename = "data_FH_dimer_ntau_$(ntau)_maxorder_$(max_order)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")

        g = h5.create_group(fid, "data")

        h5.attributes(g)["qmc_convergence_atol"] = qmc_convergence_atol
        h5.attributes(g)["ntau"] = ntau
        h5.attributes(g)["N_per_chunk"] = N_per_chunk
        h5.attributes(g)["diff_0_exa"] = diff_0_exa
        h5.attributes(g)["diff_0_nca"] = diff_0_nca
        h5.attributes(g)["diff_0_oca"] = diff_0_oca
        h5.attributes(g)["diff_0_tca"] = diff_0_tca

        g["orders"] = collect(orders)
        g["orders_bare"] = collect(orders_bare)
        g["N_chunkss"] = N_chunkss

        g["diffs_exa"] = diffs_exa
        g["diffs_nca"] = diffs_nca
        g["diffs_oca"] = diffs_oca
        g["diffs_tca"] = diffs_tca
        
        h5.close(fid)
        
    end        
end


MPI.Init()

#ntaus = 2 .^ range(4, 12)
#ntaus = 2 .^ range(3, 6)
#ntaus = 2 .^ range(3, 8)
#ntaus = 2 .^ range(9, 11)
ntaus = 2 .^ range(3, 12)
#ntaus = [128]
#ntaus = [128]

#N_chunkss = 2 .^ range(0, 7)
#N_chunkss = 2 .^ range(8, 12)
#N_chunkss = 2 .^ range(0, 12)
N_chunkss = 2 .^ range(0, 8)
#N_chunkss = [2^5]

orderss = [0:3]
#orderss = [0:2]
#orderss = [0:2, 0:3, 0:1]
#orderss = [0:1, 0:2, 0:3]
#orderss = [0:1, 0:2, 0:3]
#orderss = [0:4]

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
