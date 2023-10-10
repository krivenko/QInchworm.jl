using MPI; MPI.Init()
using HDF5; h5 = HDF5

using LinearAlgebra: diagm
using Printf
using MD5

using PyPlot; plt = PyPlot # DEBUG

using SparseArrays; sa = SparseArrays

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators

using QInchworm.expansion: Expansion, InteractionPair

using QInchworm.utility
using QInchworm.ppgf
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.inchworm: inchworm!
using QInchworm.mpi: ismaster

include("KeldyshED_addons.jl")

using .KeldyshED_addons: reduced_density_matrix,
                         density_matrix,
                         eigenstate_density_matrix,
                         reduced_ppgf,
                         occupation_number_basis_ppgf,
                         evolution_operator_imtime,
                         tofockbasis_imtime,
                         reduced_evolution_operator_imtime

function run_dimer(nτ, orders, orders_bare, N_samples; interpolate_gfs=false)

    β = 1.0
    ϵ_1, ϵ_2, V = 0.01, 0.0, 1.0

    # Combined system

    H_1 = ϵ_1 * op.n(1)
    H_2 = ϵ_2 * op.n(2)

    H_12 = H_1 + H_2 + V * (op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1))

    soi_1 = KeldyshED.Hilbert.SetOfIndices([[1]])
    soi_2 = KeldyshED.Hilbert.SetOfIndices([[2]])
    soi_12 = KeldyshED.Hilbert.SetOfIndices([[1], [2]])

    ed_1 = KeldyshED.EDCore(H_1, soi_1)
    ed_2 = KeldyshED.EDCore(H_2, soi_2)
    ed_12 = KeldyshED.EDCore(H_12, soi_12)

    # Many-body density matrices

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

    # Propagators

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    S_1 = tofockbasis_imtime(evolution_operator_imtime(ed_1, grid), ed_1)
    S_2 = tofockbasis_imtime(evolution_operator_imtime(ed_2, grid), ed_2)


    S_2_tr = deepcopy(S_2[1])
    S_2_tr.mat.data[:] = S_2[1].mat.data + S_2[2].mat.data

    S_12_red1 = reduced_evolution_operator_imtime(ed_12, soi_1, grid)

    S_12_red1_TrS2 = deepcopy(S_12_red1)

    S_12_red1_TrS2.mat.data[:] = S_12_red1.mat.data ./ S_2_tr.mat.data

    # Convert S to P
    P_1 = ppgf.atomic_ppgf(grid, ed_1)
    P_12 = ppgf.atomic_ppgf(grid, ed_12)

    P_12_red1 = reduced_ppgf(P_12, ed_12, ed_1)

    # Normalize

    PS_1 = deepcopy(P_1)
    PS_1[1].mat.data[:] = - im * S_1[1].mat.data
    PS_1[2].mat.data[:] = - im * S_1[2].mat.data
    ppgf.normalize!(PS_1, β)

    PS_12_red1_TrS2 = deepcopy(P_12_red1)
    PS_12_red1_TrS2.mat.data[1, 1, :] = - im * S_12_red1_TrS2.mat.data[1, 1, :]
    PS_12_red1_TrS2.mat.data[2, 2, :] = - im * S_12_red1_TrS2.mat.data[2, 2, :]
    ppgf.normalize!([PS_12_red1_TrS2], β)

    ρ_ps_12_red1_TrS2 = real(eigenstate_density_matrix([PS_12_red1_TrS2])[1])

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ_2), grid)

    # Pseudo Particle Strong Coupling Expansion

    if interpolate_gfs
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(reverse(Δ)))
        expansion = Expansion(ed_1, grid, [ip_fwd, ip_bwd], interpolate_ppgf=true)
    else
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
        expansion = Expansion(ed_1, grid, [ip_fwd, ip_bwd])
    end

    ρ_0 = density_matrix(expansion.P0, ed_1)

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    if interpolate_gfs
        P = [ p.GF for p in expansion.P ]
        ppgf.normalize!(P, β)
        ρ_wrm = density_matrix(P, ed_1)
    else
        ppgf.normalize!(expansion.P, β)
        ρ_wrm = density_matrix(expansion.P, ed_1)
    end

    τ = kd.imagtimes(grid)

    P1 = -imag(expansion.P[1].mat.data[1, 1, :])
    P2 = -imag(expansion.P[2].mat.data[1, 1, :])

    P1_exact = -imag(PS_12_red1_TrS2.mat.data[1, 1, :])
    P2_exact = -imag(PS_12_red1_TrS2.mat.data[2, 2, :])

    D1 = P1 - P1_exact
    D2 = P2 - P2_exact

    diff = maximum([maximum(abs.(D1)), maximum(abs.(D2))])
    max_order = maximum(orders)

    # Dump to h5 file
    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, P1)))
        filename = "data_dimer_ntau_$(nτ)_maxorder_$(max_order)_" *
                   "Nsamples_$(N_samples)_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            g = h5.create_group(fid, "data")

            h5.attributes(g)["beta"] = β
            h5.attributes(g)["V"] = V
            h5.attributes(g)["e1"] = ϵ_1
            h5.attributes(g)["e2"] = ϵ_2

            h5.attributes(g)["maxorder"] = max_order
            h5.attributes(g)["ntau"] = nτ
            h5.attributes(g)["N_samples"] = N_samples
            h5.attributes(g)["diff"] = diff

            g["orders"] = collect(orders)
            g["orders_bare"] = collect(orders_bare)

            g["tau"] = collect(τ)
            g["P1"] = P1
            g["P2"] = P2
            g["P1_exact"] = P1_exact
            g["P2_exact"] = P2_exact
        end
    end

    # Visualize
    if ismaster()

        plt.figure(figsize=(7, 10))
        subp = [4, 1, 1]

        plt.subplot(subp...); subp[end] += 1
        plt.title("nτ = $nτ, N_samples = $N_samples, orders = $orders, diff = $diff")
        plt.plot(τ, real(S_1[1].mat.data[1, 1, :]), "-", label="S_1[1]")
        plt.plot(τ, real(S_1[2].mat.data[1, 1, :]), "-", label="S_1[2]")

        plt.plot(τ, real(S_12_red1.mat.data[1, 1, :]), "-", label="S_12_red1[1,1]")
        plt.plot(τ, real(S_12_red1.mat.data[2, 2, :]), "-", label="S_12_red1[2,2]")

        plt.plot(τ, real(S_12_red1_TrS2.mat.data[1, 1, :]), "-",
                 label="S_12_red1_TrS2[1,1]")
        plt.plot(τ, real(S_12_red1_TrS2.mat.data[2, 2, :]), "-",
                 label="S_12_red1_TrS2[2,2]")

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
        plt.savefig("figure_nτ_$(nτ)_N_samples_$(N_samples)_orders_$(orders).pdf")
    end

    if ismaster()
        @show diff
    end

    return diff
end

nτs = [1024*16*2, 1024*16*4, 1024*16*8]
orderss = [0:3, 0:4]
N_sampless = collect(2 .^ (9:12))

diffs = [
    run_dimer(nτ, orders, orders, N_samples, interpolate_gfs=false)
    for N_samples in N_sampless, orders in orderss, nτ in nτs
]
