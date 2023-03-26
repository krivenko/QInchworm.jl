using MPI

using Printf
using LinearAlgebra
using MD5
using HDF5; h5 = HDF5

using PyCall

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf
using QInchworm.expansion: Expansion, InteractionPair

using QInchworm.topology_eval: get_topologies_at_order,
                               get_diagrams_at_order

using QInchworm.inchworm: inchworm_matsubara!
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.utility: inch_print


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


function run_dimer(ntau, orders, orders_bare, N_samples; interpolate_gfs=false)

    if inch_print(); @show interpolate_gfs; end

    β = 8.0
    #β = 32.0
    μ = 0.0
    #V = 1.0
    V = 0.25
    t_bethe = 1.0
    #μ_bethe = 4.0
    #μ_bethe = 2.0
    μ_bethe = 1.0
    #μ_bethe = 0.0

    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    H = μ * op.n(1)
    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H, soi)

    # -- Hybridization propagator

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

    if interpolate_gfs
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(reverse(Δ)))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=true)
    else
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])
    end

    ρ_0 = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P0), ed), ed)

    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_samples)

    if interpolate_gfs
        P = [ p.GF for p in expansion.P ]
        ppgf.normalize!(P, β) # DEBUG fixme!
        ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(P), ed), ed)
    else
        ppgf.normalize!(expansion.P, β) # DEBUG fixme!
        ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P), ed), ed)

        ρ_wrm_orders = [ full_hs_matrix(tofockbasis(ppgf.density_matrix(p), ed), ed)
                         for p in expansion.P_orders ]
        ρ_wrm_orders = [ real(diag(r)) for r in ρ_wrm_orders ]
        norm = sum(sum(ρ_wrm_orders))
        ρ_wrm_orders /= norm
        pto_hist = Array{Float64}([ sum(r) for r in ρ_wrm_orders ])

        #@show ρ_wrm_orders
        #@show norm
        #@show ρ_wrm_orders
    end

    ρ_ref = zero(ρ_wrm)

    #ρ_ref[1, 1] = 0.4755162392200716
    #ρ_ref[2, 2] = 0.5244837607799284

    #ρ_ref[1, 1] = 0.3309930890867673
    #ρ_ref[2, 2] = 0.6690069109132327

    #ρ_ref[1, 1] = 0.1763546951780647
    #ρ_ref[2, 2] = 0.8236453048219353

    ρ_ref[1, 1] = 1 - 0.5767879786180553
    ρ_ref[2, 2] = 0.5767879786180553

    diff = maximum(abs.(ρ_ref - ρ_wrm))

    if inch_print()
        @show ρ_wrm_orders
        @show pto_hist
        @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
        @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
        @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])

        @show diff
    end
    return diff, pto_hist
end

function run_ntau_calc(ntau::Integer, orders, N_sampless)

    comm_root = 0
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    orders_bare = orders

    diff_0, pto_hist_0 = run_dimer(ntau, orders, orders_bare, 0, interpolate_gfs=false)

    diffs_pto_hists = [
        run_dimer(ntau, orders, orders_bare,
                  N_samples, interpolate_gfs=false)
              for N_samples in N_sampless ]

    diffs = [ el[1] for el in diffs_pto_hists ]
    pto_hists = [ el[2] for el in diffs_pto_hists ]

    if comm_rank == comm_root
        @show diffs

        max_order = maximum(orders)
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        filename = "data_bethe_ntau_$(ntau)_maxorder_$(max_order)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")

        g = h5.create_group(fid, "data")

        h5.attributes(g)["ntau"] = ntau
        h5.attributes(g)["diff_0"] = diff_0

        g["orders"] = collect(orders)
        g["orders_bare"] = collect(orders_bare)
        g["N_sampless"] = N_sampless

        g["diffs"] = diffs
        g["pto_hists"] = reduce(hcat, pto_hists)

        h5.close(fid)
    end

    return

end

MPI.Init()

#ntaus = 2 .^ range(4, 12)
#N_samples = 8 * 2 .^ range(0, 13)
#orderss = [0:1, 0:3]

#ntaus = 2 .^ range(4, 12)
#ntaus = [64]
#N_sampless = 2 .^ range(10, 10)
#N_sampless = 2 .^ range(3, 23)
#orderss = [0:1]

#ntaus = 2 .^ range(4, 8)
#N_sampless = 2 .^ range(3, 10)
#ntaus = 2 .^ range(9, 10)
#ntaus = 2 .^ range(11, 12)
#N_sampless = 2 .^ range(3, 15)
#orderss = [0:5]
#orderss = [[0,1,3,5]]

#ntaus = [1024]
#ntaus = 2 .^ range(11, 12)
#N_sampless = 2 .^ range(3, 15)
#orderss = [0:1]
#orderss = [0:1, 0:3]

#ntaus = [128]
#N_sampless = 2 .^ range(8, 8)
#orderss = [0:5]

ntaus = [1024 * 2]
N_sampless = 2 .^ range(3, 15)
orderss = [0:5]

if inch_print()
    @show ntaus
    @show N_sampless
    @show orderss
end

#exit()

for orders in orderss
    for ntau in ntaus
        run_ntau_calc(ntau, orders, N_sampless)
    end
end
