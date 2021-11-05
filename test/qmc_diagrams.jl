
using Test

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf

import QInchworm; cfg = QInchworm.configuration

import QInchworm.configuration: Expansion, InteractionPair
import QInchworm.configuration: Configuration, Node, InchNode, NodePair, NodePairs

import QInchworm.qmc_integrate: qmc_time_ordered_integral

@testset "qmc_diagrams" begin

    # -- Single state

    β = 5.   # Inverse temperature
    U = +1.0 # Coulomb interaction
    μ = +0.5 # Chemical potential
    ϵ = +0.1 # Bath energy level
    V = -0.1 # Hybridization

    # -- Discretization

    nt = 10
    ntau = 50
    tmax = 1.

    # -- Quasi Monte Carlo
    N = 100000
    τ_qmc = 5

    # -- Exact Diagonalization

    H = - μ * (op.n(0, "up") + op.n(0, "dn")) + U * op.n(0, "up") * op.n(0, "dn")
    soi = ked.Hilbert.SetOfIndices([[0, "up"], [0, "dn"]]);
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

    # -- 3-branch time contour

    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
    grid = kd.FullTimeGrid(contour, nt, ntau);

    # -- Hybridization propagator

    Δ = kd.FullTimeGF(
        (t1, t2) -> -1.0im * V^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
        grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip_up = InteractionPair(op.c_dag(0, "up"), op.c(0, "up"), Δ)
    ip_dn = InteractionPair(op.c_dag(0, "dn"), op.c(0, "dn"), Δ)
    ppsc_exp = Expansion(ed, grid, [ip_up, ip_dn])

    # -----------------------------------------------
    # -- 2nd order inching on the imaginary branch --
    # -----------------------------------------------

    tau_grid = grid[kd.imaginary_branch]
    τ_0, τ_β = tau_grid[1], tau_grid[end]

    τ_i = τ_0
    τ_f = τ_β
    τ_w = tau_grid[end-1]

    n_i = Node(τ_i.bpoint)
    n_w = InchNode(τ_w.bpoint)
    n_f = Node(τ_f.bpoint)

    nodes = [n_f, n_w, n_i]
    conf_0 = Configuration(nodes, NodePairs())

    # -------------------------------------------------------
    # -- Quasi Monte Carlo integration over a d = 3 domain --
    # -------------------------------------------------------

    val = qmc_time_ordered_integral(3,
                                    contour,
                                    τ_i.bpoint,
                                    τ_f.bpoint,
                                    init = zero(cfg.eval(ppsc_exp, conf_0)),
                                    τ = τ_qmc,
                                    N = N) do τ
        conf = Configuration(nodes, [NodePair(n_f.time, τ[2], 1),
                                     NodePair(τ[3], τ[1], 2)])
        cfg.eval(ppsc_exp, conf)
    end

    for (s, P_s) in enumerate(ppsc_exp.P)
        sf, mat = val[s]
        ppgf.set_matsubara(P_s, τ_f, mat)
    end

    #ppgf.normalize!(ppsc_exp.P, β)
    #
    #Z = ppgf.partition_function(ppsc_exp.P)
    #λ = log(Z) / β

end
