using MPI

using Printf
using LinearAlgebra
using MD5
using HDF5; h5 = HDF5

using PyCall

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf
using QInchworm.diagrammatics: get_topologies_at_order
using QInchworm.expansion: Expansion, InteractionPair, get_diagrams_at_order

using QInchworm.inchworm: inchworm!
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.mpi: ismaster

using QuadGK: quadgk


function semi_circular_g_tau(times, t, h, β)

    g_out = zero(times)

    function kernel(t, w)
        if w > 0
            return exp(-t * w) / (1 + exp(-w))
        else
            return exp((1 - t)*w) / (1 + exp(w))
        end
    end

    for (i, τ) in enumerate(times)
        I = x -> -2 / pi / t^2 * kernel(τ/β, β*x) * sqrt(x + t - h) * sqrt(t + h - x)
        g, err = quadgk(I, -t+h, t+h; rtol=1e-12)
        g_out[i] = g
    end

    return g_out
end

function run_dimer(nτ, orders, orders_bare, N_samples; interpolate_gfs=false)

    if ismaster(); @show interpolate_gfs; end

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
    grid = kd.ImaginaryTimeGrid(contour, nτ);

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

    inchworm!(expansion,
              grid,
              orders,
              orders_bare,
              N_samples; n_pts_after_max=1)

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

    if ismaster()
        @show ρ_wrm_orders
        @show pto_hist
        @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
        @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
        @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])

        @show diff
    end
    return diff, pto_hist
end

function run_nτ_calc(nτ::Integer, orders, N_sampless)

    comm_root = 0
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    orders_bare = orders

    diff_0, pto_hist_0 = run_dimer(nτ, orders, orders_bare, 0, interpolate_gfs=false)

    diffs_pto_hists = [
        run_dimer(nτ, orders, orders_bare,
                  N_samples, interpolate_gfs=false)
              for N_samples in N_sampless ]

    diffs = [ el[1] for el in diffs_pto_hists ]
    pto_hists = [ el[2] for el in diffs_pto_hists ]

    if comm_rank == comm_root
        @show diffs

        max_order = maximum(orders)
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        filename = "data_bethe_ntau_$(nτ)_maxorder_$(max_order)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")

        g = h5.create_group(fid, "data")

        h5.attributes(g)["ntau"] = nτ
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

nτs = [1024 * 8 * 4]
N_sampless = 2 .^ (3:15)
orderss = [0:4]

if ismaster()
    @show nτs
    @show N_sampless
    @show orderss
end

#exit()

for orders in orderss
    for nτ in nτs
        run_nτ_calc(nτ, orders, N_sampless)
    end
end
