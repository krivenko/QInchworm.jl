using Test

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.configuration: Expansion, InteractionPair
import QInchworm.topology_eval: get_topologies_at_order,
                                get_diagrams_at_order
import QInchworm.inchworm: InchwormOrderData,
                           inchworm_step,
                           inchworm_step_bare,
                           inchworm_matsubara!

# -- Single state pseudo particle expansion

β = 10.
μ = +0.1 # Chemical potential
ϵ = +0.1 # Bath energy level
V = -0.1 # Hybridization
nt = 10
ntau = 20
tmax = 1.

H = - μ * op.n("0")
soi = KeldyshED.Hilbert.SetOfIndices([["0"]])
ed = KeldyshED.EDCore(H, soi)
ρ = KeldyshED.density_matrix(ed, β)

@testset "inchworm_step" begin
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
    grid = kd.FullTimeGrid(contour, nt, ntau);

    # -- Hybridization propagator

    Δ = kd.FullTimeGF(
        (t1, t2) -> -1.0im * V^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
        grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ)
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    # Fixing the initial, final and worm-time
    i_idx = 1
    f_idx = 8
    w_idx = f_idx - 1

    τ_grid = grid[kd.imaginary_branch]
    τ_i = τ_grid[i_idx].bpoint
    τ_w = τ_grid[w_idx].bpoint
    τ_f = τ_grid[f_idx].bpoint

    orders = 0:3
    N_chunk = 1000
    max_chunks = 10
    qmc_convergence_atol = 1e-3

    order_data = InchwormOrderData[]
    for order in 0:3
        topologies = get_topologies_at_order(order, 1)
        diagrams = get_diagrams_at_order(expansion, topologies, order)
        push!(order_data, InchwormOrderData(order,
                                            diagrams,
                                            N_chunk,
                                            max_chunks,
                                            qmc_convergence_atol))
    end

    value = inchworm_step(expansion, contour, τ_i, τ_w, τ_f, order_data)
    @show value
end

@testset "inchworm_step_bare" begin
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
    grid = kd.FullTimeGrid(contour, nt, ntau);

    # -- Hybridization propagator

    Δ = kd.FullTimeGF(
        (t1, t2) -> -1.0im * V^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
        grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ)
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    # Fixing the initial, final and worm-time
    i_idx = 1
    f_idx = 2

    τ_grid = grid[kd.imaginary_branch]
    τ_i = τ_grid[i_idx].bpoint
    τ_f = τ_grid[f_idx].bpoint

    orders = 0:3
    N_chunk = 1000
    max_chunks = 10
    qmc_convergence_atol = 1e-3

    order_data = InchwormOrderData[]
    for order in 0:3
        topologies = get_topologies_at_order(order)
        diagrams = get_diagrams_at_order(expansion, topologies, order)
        push!(order_data, InchwormOrderData(order,
                                            diagrams,
                                            N_chunk,
                                            max_chunks,
                                            qmc_convergence_atol))
    end

    value = inchworm_step_bare(expansion, contour, τ_i, τ_f, order_data)
    @show value
end

@testset "inchworm_matsubara" begin
    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    # -- Hybridization propagator

    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
        grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ)
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    orders = 0:3
    orders_bare = 0:2
    N_chunk = 1000
    max_chunks = 10
    qmc_convergence_atol = 1e-3

    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    @show expansion.P
end
