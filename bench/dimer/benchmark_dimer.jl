using MPI; MPI.Init()

using MD5
using HDF5; h5 = HDF5

using Printf

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf
using QInchworm.expansion: Expansion, InteractionPair

using QInchworm.inchworm: inchworm!
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.mpi: ismaster


function run_dimer(nτ, orders, orders_bare, N_samples; interpolate_gfs=false)

    β = 1.0
    ϵ_1, ϵ_2 = 0.0, 2.0
    V = 0.5

    # ED solution

    H_dimer = ϵ_1 * op.n(1) + ϵ_2 * op.n(2) +
              V * (op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1))
    soi_dimer = ked.Hilbert.SetOfIndices([[1], [2]])
    ed_dimer = ked.EDCore(H_dimer, soi_dimer)

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    H = ϵ_1 * op.n(1)
    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H, soi)

    ρ_ref = Array{ComplexF64}(reduced_density_matrix(ed_dimer, soi, β))

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ_2), grid)

    # Pseudo Particle Strong Coupling Expansion

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

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    if interpolate_gfs
        P = [ p.GF for p in expansion.P ]
        ppgf.normalize!(P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(P), ed), ed)
    else
        ppgf.normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P), ed), ed)
    end

    diff = maximum(abs.(ρ_ref - ρ_wrm))

    if ismaster()
        @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
        @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
        @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])

        @show diff
    end
    return diff
end

function run_nτ_calc(nτ, orders, N_sampless)

    orders_bare = orders

    diff_0 = run_dimer(nτ, orders, orders_bare, 0, interpolate_gfs=false)

    diffs = [run_dimer(nτ, orders, orders_bare, N_samples, interpolate_gfs=false)
             for N_samples in N_sampless]

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        filename = "data_ntau_$(nτ)_md5_$(id).h5"
        @show filename

        h5.h5open(filename, "w") do fid
            g = h5.create_group(fid, "data")

            h5.attributes(g)["ntau"] = nτ
            h5.attributes(g)["diff_0"] = diff_0

            g["orders"] = collect(orders)
            g["orders_bare"] = collect(orders_bare)
            g["N_sampless"] = N_sampless

            g["diffs"] = diffs
        end
    end
end

#nτs = 2 .^ (4:12)
#N_sampless = 8 * 2 .^ (0:13)

nτs = [1024]
N_sampless = 2 .^ (3:23)
orderss = [0:1, 0:3]

if ismaster()
    @show nτs
    @show N_sampless
    @show orderss
end

for orders in orderss
    for nτ in nτs
        run_nτ_calc(nτ, orders, N_sampless)
    end
end
