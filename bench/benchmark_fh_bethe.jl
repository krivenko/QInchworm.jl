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

function run_hubbard_dimer(ntau, orders, orders_bare, N_samples)

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
                        N_samples)

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

function run_ntau_calc(ntau::Integer, orders, N_sampless)

    comm_root = 0
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    orders_bare = orders

    # -- Do calculation here
    diffs_exa = Array{Float64}(undef, length(N_sampless))
    diffs_nca = Array{Float64}(undef, length(N_sampless))
    diffs_oca = Array{Float64}(undef, length(N_sampless))
    diffs_tca = Array{Float64}(undef, length(N_sampless))

    diff_0_exa, diff_0_nca, diff_0_oca, diff_0_tca =
        run_hubbard_dimer(ntau, orders, orders_bare, 0)
    
    for (idx, N_samples) in enumerate(N_sampless)
        diffs_exa[idx], diffs_nca[idx], diffs_oca[idx], diffs_tca[idx] =
            run_hubbard_dimer(ntau, orders, orders_bare, N_samples)
    end
        
    if comm_rank == comm_root

        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs_exa)))
        max_order = maximum(orders)
        filename = "data_FH_dimer_ntau_$(ntau)_maxorder_$(max_order)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")

        g = h5.create_group(fid, "data")

        h5.attributes(g)["ntau"] = ntau
        h5.attributes(g)["diff_0_exa"] = diff_0_exa
        h5.attributes(g)["diff_0_nca"] = diff_0_nca
        h5.attributes(g)["diff_0_oca"] = diff_0_oca
        h5.attributes(g)["diff_0_tca"] = diff_0_tca

        g["orders"] = collect(orders)
        g["orders_bare"] = collect(orders_bare)
        g["N_sampless"] = N_sampless

        g["diffs_exa"] = diffs_exa
        g["diffs_nca"] = diffs_nca
        g["diffs_oca"] = diffs_oca
        g["diffs_tca"] = diffs_tca
        
        h5.close(fid)
        
    end        
end


MPI.Init()

#ntaus = 2 .^ range(3, 12)
#ntaus = 2 .^ range(4, 8)
ntaus = 2 .^ range(4, 12)
N_sampless = 2 .^ range(4, 15)
orderss = [0:2, 0:3]

if inch_print()
    @show ntaus
    @show N_sampless
    @show orderss
end

#exit()

for ntau in ntaus
    for orders in orderss
        run_ntau_calc(ntau, orders, N_sampless)
    end
end
