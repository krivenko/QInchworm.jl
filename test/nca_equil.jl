using Test

using MPI; MPI.Init()

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.spline_gf: SplineInterpolatedGF, update_interpolants!

using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.configuration: Configuration, Node, InchNode, NodePair
using QInchworm; cfg = QInchworm.configuration;
using QInchworm.qmc_integrate: qmc_time_ordered_integral_root

import QInchworm.ppgf

function ppgf.set_matsubara!(
    g::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar} where {T, scalar},
    τ, value)
    g[τ, g.grid[1], τ_max=τ] = value
end

function ppgf.normalize!(
    g::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar} where {T, scalar},
    λ)
    τ_0 = g.grid[1]
    for τ in g.grid
        g.GF[τ, τ_0] = g[τ, τ_0] .* exp(-1im * τ.bpoint.val * λ)
    end
    update_interpolants!(g)
end

function reverse_gf(g)
    g_rev = deepcopy(g)
    τ_0, τ_β = first(g.grid), last(g.grid)
    for τ in g.grid
        g_rev[τ, τ_0] = g[τ_β, τ]
    end
    return g_rev
end

function reverse_gf(
    g::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar} where {T, scalar},
    )
    g_rev = deepcopy(g)
    τ_0, τ_β = first(g.grid), last(g.grid)
    for τ in g.grid
        g_rev.GF[τ, τ_0] = g[τ_β, τ]
    end
    return g_rev
end

@testset "nca_equil" begin

    # -- Single state

    β = 10.
    μ = +0.1 # Chemical potential
    ϵ = +0.1 # Bath energy level
    V = -0.1 # Hybridization

    # -- Discretization

    nt = 10
    ntau = 30
    tmax = 1.

    # -- Quasi Monte Carlo
    N = 2^13

    # -- Exact Diagonalization

    H = - μ * op.n("0")
    soi = ked.Hilbert.SetOfIndices([["0"]]);
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

    # -----------------------------------------------
    # -- 1st order inching on the imaginary branch --
    # -----------------------------------------------

    # -----------------------------
    # -- Riemann sum integration --
    # -----------------------------

    function run_nca_equil_tests_riemann(contour, grid, Δ, interpolate_ppgf=false)

        ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
        ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), reverse_gf(Δ))
        ppsc_exp = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=interpolate_ppgf)

        tau_grid = grid[kd.imaginary_branch]
        τ_0, τ_beta = tau_grid[1], tau_grid[end]
        Δτ = -imag(step(grid, kd.imaginary_branch))

        n_0 = Node(τ_0.bpoint)

        for (fidx, τ_f) in enumerate(tau_grid[2:end])

            τ_w = tau_grid[fidx]

            n_f = Node(τ_f.bpoint)
            n_w = InchNode(τ_w.bpoint)
            nodes = [n_f, n_w, n_0]

            conf_0 = Configuration(nodes, NodePair[], ppsc_exp)

            val = cfg.eval(ppsc_exp, conf_0)

            for τ_1 in tau_grid[1:fidx]

                p_fwd = NodePair(n_f.time, τ_1.bpoint, 1)
                conf_1_fwd = Configuration(nodes, [p_fwd], ppsc_exp)
                val -= Δτ^2 * cfg.eval(ppsc_exp, conf_1_fwd)

                p_bwd = NodePair(n_f.time, τ_1.bpoint, 2)
                conf_1_bwd = Configuration(nodes, [p_bwd], ppsc_exp)
                val -= Δτ^2 * cfg.eval(ppsc_exp, conf_1_bwd)

            end

            for (s, P_s) in enumerate(ppsc_exp.P)
                sf, mat = val[s]
                ppgf.set_matsubara!(P_s, τ_f, mat)
            end

        end

        ppgf.normalize!(ppsc_exp.P, β)

        Z = ppgf.partition_function(ppsc_exp.P)
        @test Z ≈ 1.0
        λ = log(Z) / β

        # ---------------------
        # -- Regression test --
        # ---------------------

        @test ppsc_exp.P[1][τ_beta, τ_0] + ppsc_exp.P[2][τ_beta, τ_0] ≈ [-im]
        @test ppsc_exp.P[1][τ_beta, τ_0][1, 1] ≈ 0.0 - 0.7127769872093338im
        @test ppsc_exp.P[2][τ_beta, τ_0][1, 1] ≈ 0.0 - 0.2872230127906664im

        interpolate_ppgf && return

        # -- Single particle Green's function

        g = ppgf.first_order_spgf(ppsc_exp.P, ed, 1, 1)

        # ---------------------------------------
        # -- Reference calculation of the spgf --
        # ---------------------------------------

        H_ref = - μ * op.n("0") + ϵ * op.n("1") +
                V * op.c_dag("1") * op.c("0") + V * op.c_dag("0") * op.c("1")
        soi_ref = ked.Hilbert.SetOfIndices([["0"], ["1"]]);
        ed_ref = ked.EDCore(H_ref, soi_ref)

        idx = ked.Hilbert.IndicesType(["0"])
        g_ref = ked.computegf(ed_ref, grid, idx, idx);

        @test isapprox(g[:matsubara], g_ref[:matsubara], atol=0.05)
    end

    # -----------------------------------
    # -- Quasi Monte Carlo integration --
    # -----------------------------------

    function run_nca_equil_tests_qmc(contour, grid, Δ, interpolate_ppgf=false)

        ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
        ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), reverse_gf(Δ))
        ppsc_exp = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=interpolate_ppgf)

        tau_grid = grid[kd.imaginary_branch]
        τ_0, τ_beta = tau_grid[1], tau_grid[end]

        n_0 = Node(τ_0.bpoint)

        for (fidx, τ_f) in enumerate(tau_grid[2:end])

            τ_w = tau_grid[fidx]

            n_f = Node(τ_f.bpoint)
            n_w = InchNode(τ_w.bpoint)
            nodes = [n_f, n_w, n_0]

            conf_0 = Configuration(nodes, NodePair[], ppsc_exp)

            val = cfg.eval(ppsc_exp, conf_0)

            # The 'im' prefactor accounts for the direction of the imaginary branch
            val -= im * qmc_time_ordered_integral_root(
                1,
                contour,
                τ_0.bpoint,
                τ_f.bpoint,
                init = zero(val),
                N = N) do τ_1
                conf_1_fwd = Configuration(nodes, [NodePair(n_f.time, τ_1[1], 1)], ppsc_exp)
                conf_1_bwd = Configuration(nodes, [NodePair(n_f.time, τ_1[1], 2)], ppsc_exp)
                cfg.eval(ppsc_exp, conf_1_fwd) + cfg.eval(ppsc_exp, conf_1_bwd)
            end

            for (s, P_s) in enumerate(ppsc_exp.P)
                sf, mat = val[s]
                ppgf.set_matsubara!(P_s, τ_f, mat)
            end

        end

        ppgf.normalize!(ppsc_exp.P, β)

        # ---------------------
        # -- Regression test --
        # ---------------------

        @test ppsc_exp.P[1][τ_beta, τ_0] + ppsc_exp.P[2][τ_beta, τ_0] ≈ [-im]
        @test isapprox(ppsc_exp.P[1][τ_beta, τ_0][1, 1],
                    0.0 - 0.6937776444966494im,
                    atol=1e-3)
        @test isapprox(ppsc_exp.P[2][τ_beta, τ_0][1, 1],
                    0.0 - 0.30622235550335064im,
                    atol=1e-3)
    end

    @testset "Imaginary time" begin
        contour = kd.ImaginaryContour(β=β)
        grid = kd.ImaginaryTimeGrid(contour, ntau)

        # -- Hybridization propagator

        Δ = kd.ImaginaryTimeGF(
            (t1, t2) -> -1.0im * V^2 *
                (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
                exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
            ComplexF64, grid, 1, kd.fermionic, true)

        run_nca_equil_tests_riemann(contour, grid, Δ, false)
        run_nca_equil_tests_qmc(contour, grid, Δ, false)

        # #############################################################
        # NOT WORKING! Fix me, when going to higher order interpolants.
        # #############################################################

        #run_nca_equil_tests_riemann(contour, grid, SplineInterpolatedGF(Δ), true)
        #run_nca_equil_tests_qmc(contour, grid, SplineInterpolatedGF(Δ), true)
    end

    # ####################################################
    # NOT WORKING! Fix me, when generalizing to real time.
    # ####################################################

    @testset "Twisted Kadanoff-Baym-Keldysh contour" begin
        contour = kd.twist(kd.FullContour(tmax=tmax, β=β))
        grid = kd.FullTimeGrid(contour, nt, ntau)

        # -- Hybridization propagator

        Δ = kd.FullTimeGF(
            (t1, t2) -> -1.0im * V^2 *
                (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
                exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
            grid, 1, kd.fermionic, true)

        #run_nca_equil_tests_riemann(contour, grid, Δ)
        #run_nca_equil_tests_qmc(contour, grid, Δ)
    end

end
