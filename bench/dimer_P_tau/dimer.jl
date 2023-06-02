# TODO: What do we do with this file? Is it sill used?

using MPI; MPI.Init()

using LinearAlgebra: diagm

using Printf

using MD5
using HDF5; h5 = HDF5

using SparseArrays; sa = SparseArrays

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.topology_eval: get_topologies_at_order,
                               get_diagrams_at_order

using QInchworm.inchworm: inchworm!
using QInchworm.ppgf
using QInchworm.KeldyshED_addons: reduced_density_matrix,
                                  density_matrix,
                                  eigenstate_density_matrix,
                                  reduced_ppgf,
                                  occupation_number_basis_ppgf,
                                  evolution_operator_imtime,
                                  tofockbasis_imtime,
                                  reduced_evolution_operator_imtime

                                  using QInchworm.spline_gf: SplineInterpolatedGF

using QInchworm.mpi: ismaster

using PyPlot; plt = PyPlot # DEBUG


function run_dimer(ntau, orders, orders_bare, N_samples; interpolate_gfs=false)

    #@time begin

    #β = 2.0
    #β = 4.0
    β = 1.0
    #ϵ_1, ϵ_2, V = 0.5, 2.0, 0.5
    #ϵ_1, ϵ_2, V = 0.0, 0.0, 1.0
    #ϵ_1, ϵ_2, V = 0.0, 0.0, 0.5
    ϵ_1, ϵ_2, V = 0.01, 0.0, 1.0

    # -- Combined system

    H_1 = ϵ_1 * op.n(1)
    H_2 = ϵ_2 * op.n(2)

    H_12 = H_1 + H_2 + V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )

    soi_1 = KeldyshED.Hilbert.SetOfIndices([[1]])
    soi_2 = KeldyshED.Hilbert.SetOfIndices([[2]])
    soi_12 = KeldyshED.Hilbert.SetOfIndices([[1], [2]])

    ed_1 = KeldyshED.EDCore(H_1, soi_1)
    ed_2 = KeldyshED.EDCore(H_2, soi_2)
    ed_12 = KeldyshED.EDCore(H_12, soi_12)

    #end
    #println("KED")

    #@time begin
    # -- Many body density matrices

    # The ked.tofockbasis returns a list of block matrices for each symmetry sector?

    # To be comparable with the reduced_density_matrix it would be
    # easier with a matrix in the entire Hilbert space?

    blockmat = m -> Matrix(sa.blockdiag(sa.sparse.(m)...))

    ρ_1 = blockmat(ked.tofockbasis(ked.density_matrix(ed_1, β), ed_1))
    ρ_2 = blockmat(ked.tofockbasis(ked.density_matrix(ed_2, β), ed_2))
    ρ_12 = blockmat(ked.tofockbasis(ked.density_matrix(ed_12, β), ed_12))

    ρ_12_red1 = ked.reduced_density_matrix(ed_12, soi_1, β)
    ρ_12_red2 = ked.reduced_density_matrix(ed_12, soi_2, β)

    if ismaster()
        @show ρ_12
        @show ρ_1
        @show ρ_2

        @show ρ_12_red1
        @show ρ_12_red2
    end


    #end
    #println("Dens mat")

    #@test ρ_12_red1 ≈ ρ_1
    #@test ρ_12_red2 ≈ ρ_2

    # -- Propagators

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    #@time begin

    #S_1 = ked.tofockbasis(ked.evolution_operator(ed_1, grid), ed_1)
    #S_2 = ked.tofockbasis(ked.evolution_operator(ed_2, grid), ed_2)

    S_1 = tofockbasis_imtime(evolution_operator_imtime(ed_1, grid), ed_1)
    S_2 = tofockbasis_imtime(evolution_operator_imtime(ed_2, grid), ed_2)

    #S_2_tr = S_2[1] + S_2[2] # Does not compile

    S_2_tr = deepcopy(S_2[1])
    #S_2_tr.data.data[:] = S_2[1].data.data + S_2[2].data.data
    S_2_tr.mat.data[:] = S_2[1].mat.data + S_2[2].mat.data
    #S_2_tr.data.data[:] = sum([ S_2[i].data.data for i in length(S_2) ], dims=1)

    #S_12_red1 = ked.reduced_evolution_operator(ed_12, soi_1, grid)
    S_12_red1 = reduced_evolution_operator_imtime(ed_12, soi_1, grid)

    #@show size(S_12_red1.data.data)
    #@show size(S_2_tr.data.data)

    S_12_red1_TrS2 = deepcopy(S_12_red1)
    #S_12_red1_TrS2.data.data[:] = S_12_red1.data.data ./ S_2_tr.data.data

    S_12_red1_TrS2.mat.data[:] = S_12_red1.mat.data ./ S_2_tr.mat.data

    #end
    #println("KED ev op")

    #@time begin
    # -- Convert S to P
    P_1 = ppgf.atomic_ppgf(grid, ed_1)
    P_12 = ppgf.atomic_ppgf(grid, ed_12)
    #end
    #println("QInch atomic prop")

    #@time begin
    P_12_red1 = reduced_ppgf(P_12, ed_12, ed_1)
    #end
    #println("QInch reduced ppgf")

    #@time begin
    # -- Normalize

    PS_1 = deepcopy(P_1)
    #PS_1[1].mat.data[1, 1, :] = - im * S_1[1].data[1, 1, :, 1]
    #PS_1[2].mat.data[1, 1, :] = - im * S_1[2].data[1, 1, :, 1]
    PS_1[1].mat.data[:] = - im * S_1[1].mat.data
    PS_1[2].mat.data[:] = - im * S_1[2].mat.data
    ppgf.normalize!(PS_1, β)

    PS_12_red1_TrS2 = deepcopy(P_12_red1)
    #PS_12_red1_TrS2.mat.data[1, 1, :] = - im * S_12_red1_TrS2.data[1, 1, :, 1]
    #PS_12_red1_TrS2.mat.data[2, 2, :] = - im * S_12_red1_TrS2.data[2, 2, :, 1]
    PS_12_red1_TrS2.mat.data[1, 1, :] = - im * S_12_red1_TrS2.mat.data[1, 1, :]
    PS_12_red1_TrS2.mat.data[2, 2, :] = - im * S_12_red1_TrS2.mat.data[2, 2, :]
    ppgf.normalize!([PS_12_red1_TrS2], β)

    ρ_ps_12_red1_TrS2 = real(eigenstate_density_matrix([PS_12_red1_TrS2])[1])
    if ismaster()
        @show ρ_ps_12_red1_TrS2
    end

    #end
    #println("QInch ev op")

    #exit()
    #@test ρ_ps_12_red1_TrS2 ≈ ρ_12_red1

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
        expansion = Expansion(ed_1, grid, [ip_fwd, ip_bwd], interpolate_ppgf=true)
        println("Using spline GFS")
    else
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
        expansion = Expansion(ed_1, grid, [ip_fwd, ip_bwd])
    end

    ρ_0 = density_matrix(expansion.P0, ed_1)

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    if interpolate_gfs
        P = [ p.GF for p in expansion.P ]
        ppgf.normalize!(P, β) # DEBUG fixme!
        ρ_wrm = density_matrix(P, ed_1)
    else
        ppgf.normalize!(expansion.P, β) # DEBUG fixme!
        ρ_wrm = density_matrix(expansion.P, ed_1)
    end

    τ = kd.imagtimes(grid)

    P1 = -imag(expansion.P[1].mat.data[1, 1, :])
    P2 = -imag(expansion.P[2].mat.data[1, 1, :])

    P1_exact = -imag(PS_12_red1_TrS2.mat.data[1, 1, :])
    P2_exact = -imag(PS_12_red1_TrS2.mat.data[2, 2, :])

    #D1 = - imag(PS_12_red1_TrS2.mat.data[1, 1, :]) + imag(expansion.P[1].mat.data[1, 1, :])
    #D2 = -imag(PS_12_red1_TrS2.mat.data[2, 2, :]) + imag(expansion.P[2].mat.data[1, 1, :])

    D1 = P1 - P1_exact
    D2 = P2 - P2_exact

    diff = maximum([maximum(abs.(D1)), maximum(abs.(D2))])
    max_order = maximum(orders)

    # -- Dump to h5 file

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, P1)))
        filename = "data_dimer_ntau_$(ntau)_maxorder_$(max_order)_Nsamples_$(N_samples)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")

        g = h5.create_group(fid, "data")

        h5.attributes(g)["beta"] = β
        h5.attributes(g)["V"] = V
        h5.attributes(g)["e1"] = ϵ_1
        h5.attributes(g)["e2"] = ϵ_2

        h5.attributes(g)["maxorder"] = max_order
        h5.attributes(g)["ntau"] = ntau
        h5.attributes(g)["N_samples"] = N_samples
        h5.attributes(g)["diff"] = diff

        g["orders"] = collect(orders)
        g["orders_bare"] = collect(orders_bare)

        g["tau"] = collect(τ)
        g["P1"] = P1
        g["P2"] = P2
        g["P1_exact"] = P1_exact
        g["P2_exact"] = P2_exact

        h5.close(fid)
    end

    if ismaster()
        @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
        #@printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
        @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])
        #diff = maximum(abs.(ρ_ref - ρ_wrm))
        #@show diff
    end


    if ismaster()
    #if true # -----  VIZ

    # -- Visualize

    plt.figure(figsize=(7, 10))

    #subp = [3, 2, 1]
    subp = [4, 1, 1]

    plt.subplot(subp...); subp[end] += 1
    plt.title("ntau = $ntau, N_samples = $N_samples, ordes = $orders, diff = $diff")
    #plt.plot(τ, real(S_1[1].data[1, 1, :, 1]), "-", label="S_1[1]")
    #plt.plot(τ, real(S_1[2].data[1, 1, :, 1]), "-", label="S_1[2]")
    plt.plot(τ, real(S_1[1].mat.data[1, 1, :]), "-", label="S_1[1]")
    plt.plot(τ, real(S_1[2].mat.data[1, 1, :]), "-", label="S_1[2]")

    #plt.plot(τ, real(S_2[1].data[1, 1, :, 1]), label="S_2[1]")
    #plt.plot(τ, real(S_2[2].data[1, 1, :, 1]), label="S_2[2]")

    #plt.plot(τ, real(S_12_red1.data[1, 1, :, 1]), "-", label="S_12_red1[1,1]")
    #plt.plot(τ, real(S_12_red1.data[2, 2, :, 1]), "-", label="S_12_red1[2,2]")
    plt.plot(τ, real(S_12_red1.mat.data[1, 1, :]), "-", label="S_12_red1[1,1]")
    plt.plot(τ, real(S_12_red1.mat.data[2, 2, :]), "-", label="S_12_red1[2,2]")

    #plt.plot(τ, real(S_12_red1_TrS2.data[1, 1, :, 1]), "-", label="S_12_red1_TrS2[1,1]")
    #plt.plot(τ, real(S_12_red1_TrS2.data[2, 2, :, 1]), "-", label="S_12_red1_TrS2[2,2]")
    plt.plot(τ, real(S_12_red1_TrS2.mat.data[1, 1, :]), "-", label="S_12_red1_TrS2[1,1]")
    plt.plot(τ, real(S_12_red1_TrS2.mat.data[2, 2, :]), "-", label="S_12_red1_TrS2[2,2]")

    plt.legend()
    plt.grid(true)

    plt.subplot(subp...); subp[end] += 1

    plt.plot(τ, -imag(P_1[1].mat.data[1, 1, :]), "-", label="P_1[1]")
    plt.plot(τ, -imag(P_1[2].mat.data[1, 1, :]), "-", label="P_1[2]")

    plt.plot(τ, -imag(PS_1[1].mat.data[1, 1, :]), "--", label="PS_1[1]")
    plt.plot(τ, -imag(PS_1[2].mat.data[1, 1, :]), "--", label="PS_1[2]")

    plt.legend()
    plt.grid(true)

    plt.subplot(subp...); subp[end] += 1

    plt.plot(τ, -imag(PS_12_red1_TrS2.mat.data[1, 1, :]),
             "--", label="PS_12_red1_TrS2[1,1]")
    plt.plot(τ, -imag(PS_12_red1_TrS2.mat.data[2, 2, :]),
             "--", label="PS_12_red1_TrS2[2,2]")

    plt.plot(τ, -imag(expansion.P[1].mat.data[1, 1, :]), "-", label="exp.P[1]")
    plt.plot(τ, -imag(expansion.P[2].mat.data[1, 1, :]), "-", label="exp.P[2]")

    plt.legend()
    plt.grid(true)

    plt.subplot(subp...); subp[end] += 1

    plt.plot(τ, D1, "-", label="D1")
    plt.plot(τ, D2, "-", label="D2")

    plt.legend()
    plt.grid(true)

    plt.tight_layout()
    plt.savefig("figure_ntau_$(ntau)_N_samples_$(N_samples)_orders_$(orders).pdf")
    #plt.show()

    end # -----  VIZ

    if ismaster()
        @show diff
    end

    return diff
end


#ntau = 128
#ntau = 256
#ntau = 512
#ntau = 1024
#ntau = 1024 * 2
#ntaus = [1024*2, 1024*4, 1024*8]
#ntaus = [1024*2*8, 1024*4*8, 1024*8*8]
ntaus = [1024*16*2, 1024*16*4, 1024*16*8]
#ntaus = [256, 512, 1024]
#ntau = 1024 * 4
#ntau = 1024 * 4 * 2 * 2
#ntau = 1024 * 4 * 2 * 2 * 2
#ntau = 32768
#ntau = 65536
#N_samples = 2^7
#N_samples = 2^10
#N_samples = 2^14

#orders = 0:2
#orderss = [0:1, 0:3, 0:4, 0:5]
orderss = [0:3, 0:4]
#orderss = [0:3, 0:4, 0:5]
#orderss = [0:4, 0:5]
#orderss = [0:4]
#orderss = [0:5]
#orderss = [0:6]
#orderss = [0:0]
#N_sampless = collect( 2 .^ (2:8) )
N_sampless = collect( 2 .^ (9:12) )
#println(N_sampless)
#exit()
#N_sampless = [2^2, 2^3, 2^4]

diffs = [
    run_dimer(ntau, orders, orders, N_samples, interpolate_gfs=false)
    for N_samples in N_sampless, orders in orderss, ntau in ntaus ]

exit()

#orderss = [0:0, 0:1, 0:3, 0:4]
#orderss = [0:0, 0:1, 0:3, 0:4, 0:5]
#orderss = [0:0, 0:1, 0:3, 0:4, 0:5, 0:6]
orderss = [0:0]

diffs = [
    run_dimer(ntau, orders, orders, N_samples, interpolate_gfs=false)
    for orders in orderss ]

if ismaster()
    @show diffs
    ord = [ maximum(o) for o in orderss ]

    plt.figure(figsize=(6, 6))
    plt.title("ntau = $ntau, N_samples = $N_samples")
    plt.plot(ord, diffs, "o-")
    plt.semilogy([], [])
    plt.xlabel("Max order")
    plt.ylabel(raw"$\max|ΔP(τ)|$")
    plt.grid(true)
    plt.tight_layout()
    #plt.legend(loc="best")
    plt.savefig("figure_dimer_convergence_ntau_$(ntau)_N_samples_$(N_samples).pdf")
end
