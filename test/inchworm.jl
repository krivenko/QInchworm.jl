using MPI; MPI.Init()

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.spline_gf: SplineInterpolatedGF

using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.topology_eval: get_topologies_at_order,
                               get_diagrams_at_order,
                               get_configurations_and_diagrams

using QInchworm.inchworm: ExpansionOrderInputData,
                          inchworm_step,
                          inchworm_step_bare,
                          inchworm_matsubara!,
                          compute_gf_matsubara

# -- Single state pseudo particle expansion

β = 10.
μ = +0.1 # Chemical potential
ϵ = +0.1 # Bath energy level
V = -0.1 # Hybridization
nt = 10
ntau = 20
tmax = 1.

H = - μ * op.n("0")
soi = ked.Hilbert.SetOfIndices([["0"]])
ed = ked.EDCore(H, soi)
ρ = ked.density_matrix(ed, β)

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
    τ_i = τ_grid[i_idx]
    τ_w = τ_grid[w_idx]
    τ_f = τ_grid[f_idx]

    orders = 0:3
    N_samples = 2^8

    # Extend expansion.P_orders to max of orders
    for o in range(1, maximum(orders)+1)
        push!(expansion.P_orders, kd.zero(expansion.P0))
    end

    order_data = ExpansionOrderInputData[]
    for order in 0:3
        for n_pts_after = 1:max(1, 2*order-1)
            d_before = 2 * order - n_pts_after
            topologies = get_topologies_at_order(order, 1)
            all_diagrams = get_diagrams_at_order(expansion, topologies, order)
            configurations, diagrams = get_configurations_and_diagrams(
                expansion, all_diagrams, d_before)
            if length(configurations) > 0
                push!(order_data, ExpansionOrderInputData(order,
                                                          n_pts_after,
                                                          diagrams,
                                                          configurations,
                                                          N_samples))
            end
        end
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
    τ_i = τ_grid[i_idx]
    τ_f = τ_grid[f_idx]

    orders = 0:3
    N_samples = 2^8

    # Extend expansion.P_orders to max of orders
    for o in range(1, maximum(orders)+1)
        push!(expansion.P_orders, kd.zero(expansion.P0))
    end

    order_data = ExpansionOrderInputData[]
    for order in 0:3
        topologies = get_topologies_at_order(order)
        all_diagrams = get_diagrams_at_order(expansion, topologies, order)
        configurations, diagrams = get_configurations_and_diagrams(
            expansion, all_diagrams, 0)
        if length(configurations) > 0
            push!(order_data, ExpansionOrderInputData(order,
                                                      1,
                                                      diagrams,
                                                      configurations,
                                                      N_samples))
        end
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

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), SplineInterpolatedGF(Δ))
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), SplineInterpolatedGF(Δ))
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf = true)

    orders = 0:3
    orders_bare = 0:2
    N_samples = 2^8

    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_samples)

    @show expansion.P

    # -- Single-particle GF

    push!(expansion.corr_operators, (op.c("0"), op.c_dag("0")))
    g = compute_gf_matsubara(expansion, grid, orders, N_samples)

    @show g
end
