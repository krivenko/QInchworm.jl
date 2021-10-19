
using Test

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf

import QInchworm; cfg = QInchworm.configuration

import QInchworm.configuration: Expansion, InteractionPair
import QInchworm.configuration: Configuration, Node, InchNode, NodePair, NodePairs


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

    # -- Exact Diagonalization

    H = - μ * op.n("0")
    soi = KeldyshED.Hilbert.SetOfIndices([["0"]]);
    ed = KeldyshED.EDCore(H, soi)
    ρ = KeldyshED.density_matrix(ed, β)

    # -- Real-time Kadanoff-Baym contour

    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
    grid = kd.FullTimeGrid(contour, nt, ntau);

    # -- Hybridization propagator

    Δ = kd.FullTimeGF(
        (t1, t2) -> -1.0im * V^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
        grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ppsc_exp = Expansion(ed, grid, [ip])

    # -----------------------
    # -- 1st order inching --
    # -----------------------

    tau_grid = grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    τ_beta = tau_grid[end]

    Δτ = -imag(tau_grid[2].bpoint.val - tau_grid[1].bpoint.val)

    n_0 = Node(τ_0.bpoint)

    for (fidx, τ_f) in enumerate(tau_grid[2:end])

        τ_w = tau_grid[fidx]

        n_f = Node(τ_f.bpoint)
        n_w = InchNode(τ_w.bpoint)
        nodes = [n_f, n_w, n_0]

        conf_0 = Configuration(nodes, NodePairs())
        val = cfg.eval(ppsc_exp, conf_0)

        for τ_1 in tau_grid[1:fidx]

            p_fwd = NodePair(n_f.time, τ_1.bpoint, 1)
            conf_1_fwd = Configuration(nodes, [p_fwd])
            val += Δτ^2 * cfg.eval(ppsc_exp, conf_1_fwd)

            p_bwd = NodePair(τ_1.bpoint, n_f.time, 1)
            conf_1_bwd = Configuration(nodes, [p_bwd])
            val += Δτ^2 * cfg.eval(ppsc_exp, conf_1_bwd)

        end

        for (s, P_s) in enumerate(ppsc_exp.P)
            sf, mat = val[s]
            ppgf.set_matsubara(P_s, τ_f, mat)
        end

    end

    Z = ppgf.partition_function(ppsc_exp.P)
    λ = log(Z) / β

    ppgf.normalize!(ppsc_exp.P, β)

    # ---------------------
    # -- Regression test --
    # ---------------------

    τ_i = grid[kd.imaginary_branch][1]
    τ_f = grid[kd.imaginary_branch][end]

    @show ppsc_exp.P[1][τ_f, τ_i] + ppsc_exp.P[2][τ_f, τ_i]
    @show ppsc_exp.P[1][τ_f, τ_i]
    @show ppsc_exp.P[2][τ_f, τ_i]

    @test ppsc_exp.P[1][τ_f, τ_i] + ppsc_exp.P[2][τ_f, τ_i] ≈ [-im]
    @test ppsc_exp.P[1][τ_f, τ_i][1, 1] ≈ 0.0 - 0.7105404445143371im
    @test ppsc_exp.P[2][τ_f, τ_i][1, 1] ≈ 0.0 - 0.289459555485663im

    # -- Single particle Green's function

    g = ppgf.first_order_spgf(ppsc_exp.P, ed, 1, 1)

    # ---------------------------------------
    # -- Reference calculation of the spgf --
    # ---------------------------------------

    # Note: For a single fermionic state NCA is exact.

    # Hence, the NCA approximation for the single particle propagator is
    # (up to discretization errors) be equal to the ED Green's function
    # of the 0th site of the two-fermion system.

    H_ref = - μ * op.n("0") + V * op.c_dag("1") * op.c("0") + V * op.c_dag("0") * op.c("1") + ϵ * op.n("1")
    soi_ref = KeldyshED.Hilbert.SetOfIndices([["0"], ["1"]]);
    ed_ref = KeldyshED.EDCore(H_ref, soi_ref)

    idx = KeldyshED.Hilbert.IndicesType(["0"])
    g_ref = KeldyshED.computegf(ed_ref, grid, idx, idx);

    diff = maximum(abs.((g[:matsubara] - g_ref[:matsubara])/g_ref[:matsubara]))
    @show diff

    @test isapprox(g[:matsubara], g_ref[:matsubara], atol=0.05)

end
