""" Solve non-interacting two fermion AIM coupled to
semi-circular (Bethe lattice) hybridization functions.

Compare to numerically exact results for the 1st, 2nd and 3rd
order dressed self-consistent expansion for the many-body
density matrix (computed using DLR elsewhere).

Note that the 1, 2, 3 order density matrix differs from the
exact density matrix of the non-interacting system, since
the low order expansions introduce "artificial" effective
interactions between hybridization insertions.

Author: Hugo U. R. Strand (2023)

"""

using MPI; MPI.Init()
using HDF5; h5 = HDF5

using LinearAlgebra: diag

using QuadGK: quadgk

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.inchworm: inchworm!
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.utility: inch_print

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
    n = 0.5460872495307262
    return ρ_from_n_ref(ρ_wrm, n)
end

function get_ρ_nca(ρ_wrm)
    rho_nca = [0.1961713995875524, 0.2474226001525296, 0.2474226001525296, 0.3089834001073883]
    return ρ_from_ρ_ref(ρ_wrm , rho_nca)
end

function get_ρ_oca(ρ_wrm)
    rho_oca = [0.2018070389569783, 0.2476929924482211, 0.2476929924482211, 0.3028069761465793]
    return ρ_from_ρ_ref(ρ_wrm , rho_oca)
end

function get_ρ_tca(ρ_wrm)
    rho_tca = [0.205163794520457, 0.2478638876741985, 0.2478638876741985, 0.2991084301311462]
    return ρ_from_ρ_ref(ρ_wrm , rho_tca)
end

function reverse(g::kd.ImaginaryTimeGF)
    g_rev = deepcopy(g)
    τ_0, τ_β = first(g.grid), last(g.grid)
    for τ in g.grid
        g_rev[τ, τ_0] = g[τ_β, τ]
    end
    return g_rev
end

function run_hubbard_dimer(ntau, orders, orders_bare, N_samples, μ_bethe, interpolation)

    β = 10.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0

    # -- ED solution

    H_imp = -μ * (op.n(1) + op.n(2))

    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    # -- Hybridization propagator

    tau = [ real(im * τ.bpoint.val) for τ in grid ]
    delta_bethe = V^2 * semi_circular_g_tau(tau, t_bethe, μ_bethe, β)

    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> 1.0im * V^2 *
            semi_circular_g_tau(
                [-imag(t1.bpoint.val - t2.bpoint.val)],
                t_bethe, μ_bethe, β)[1],
        grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    if interpolation
        ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
        ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(reverse(Δ)))
        ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), SplineInterpolatedGF(Δ))
        ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), SplineInterpolatedGF(reverse(Δ)))
        expansion = Expansion(ed,
                              grid,
                              [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd],
                              interpolate_ppgf=true)
    else
        ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
        ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
        ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))
        expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])
    end

    ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    if interpolation
        # Extract a plain (non-interpolated) version of expansion.P
        P = [deepcopy(P_int.GF) for P_int in expansion.P]

        normalize!(P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(P), ed), ed)
    else
        normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(expansion.P), ed), ed)
    end

    ρ_exa = get_ρ_exact(ρ_wrm)
    ρ_nca = get_ρ_nca(ρ_wrm)
    ρ_oca = get_ρ_oca(ρ_wrm)
    ρ_tca = get_ρ_tca(ρ_wrm)

    diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
    diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
    diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
    diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

    ρ_000 = real(diag(ρ_0))
    ρ_exa = real(diag(ρ_exa))
    ρ_nca = real(diag(ρ_nca))
    ρ_oca = real(diag(ρ_oca))
    ρ_tca = real(diag(ρ_tca))
    ρ_wrm = real(diag(ρ_wrm))

    if comm_rank == 0
        @show ρ_wrm
        @show sum(ρ_wrm)

        @show ρ_000
        @show ρ_nca
        @show ρ_oca
        @show ρ_tca
        @show ρ_exa

        @show diff_nca
        @show diff_oca
        @show diff_tca
        @show diff_exa
    end

    ρ_ref = [ρ_000, ρ_nca, ρ_oca, ρ_tca, ρ_exa]
    diffs = [diff_nca, diff_oca, diff_tca, diff_exa]

    return ρ_wrm, ρ_ref, diffs
end

comm = MPI.COMM_WORLD
comm_size = MPI.Comm_size(comm)
comm_rank = MPI.Comm_rank(comm)

μ_bethe = 0.25
N_samples = 2^12
nτ_list = 2 .^ range(3, 10)

if comm_rank == 0
    fid = h5.h5open("data_inchworm_interp.h5", "w")
    data_g = h5.create_group(fid, "data")
    h5.attributes(data_g)["mu_bethe"] = μ_bethe
    h5.attributes(data_g)["N_samples"] = N_samples
end

for orders in [0:1, 0:2, 0:3]

    diffs_non_interp = zeros(Float64, length(nτ_list), 4)
    diffs_interp = zeros(Float64, length(nτ_list), 4)

    for (i, nτ) in pairs(nτ_list)
        ρ_wrm, ρ_ref, diffs =
            run_hubbard_dimer(nτ, orders, orders, N_samples, μ_bethe, false)
        diffs_non_interp[i, :] = diffs
        ρ_wrm, ρ_ref, diffs =
            run_hubbard_dimer(nτ, orders, orders, N_samples, μ_bethe, true)
        diffs_interp[i, :] = diffs
    end

    if comm_rank == 0
        println("orders = ", orders)
        print("diffs_non_interp = ")
        display(diffs_non_interp)
        print("diffs_interp = ")
        display(diffs_interp)

        g = h5.create_group(data_g, "orders_$orders")
        h5.attributes(g)["ntau_list"] = nτ_list
        g["orders"] = collect(orders)
        g["diffs_non_interp"] = diffs_non_interp
        g["diffs_interp"] = diffs_interp
    end
end

if comm_rank == 0
    h5.close(fid)
end
