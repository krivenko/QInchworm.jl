using Test
using Random

using MPI; MPI.Init()

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm
cfg = QInchworm.configuration
teval = QInchworm.topology_eval

using QInchworm.expansion: Expansion, InteractionPair

@testset "topology_to_config" begin

    # -- Single state pseudo particle expansion

    β = 10.
    μ = +0.1 # Chemical potential
    ϵ = +0.1 # Bath energy level
    V = -0.1 # Hybridization
    nt = 10
    ntau = 30
    tmax = 1.

    H = - μ * op.n("0")
    soi = ked.Hilbert.SetOfIndices([["0"]])
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

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

    # -- Inch-worm node configuration, fixing the final-time and worm-time

    fidx = 8
    widx = fidx - 1

    worm_nodes = teval.get_imaginary_time_worm_nodes(grid, fidx, widx)

    # -- Generate all topologies and diagrams at `order`

    order = 3
    topologies = teval.get_topologies_at_order(order, 1)
    diagrams = teval.get_diagrams_at_order(expansion, topologies, order)

    @test length(diagrams) == length(topologies) * length(expansion.pairs)^order

    println("n_topologies = $(length(topologies))")
    for topology in topologies
        println(topology)
    end

    println("n_diagrams = $(length(diagrams))")
    for diagram in diagrams
        println(diagram)
    end

    accumulated_value = 0 * cfg.operator(expansion, first(worm_nodes))

    n_samples = 100

    for sample in range(1, n_samples)

        # -- Generate time ordered points on the unit-interval (replace with quasi-MC points)
        # -- separating the initial point `x1` (between the final- and worm-time) from the others `xs`

        x1 = Random.rand(Float64)
        xs = Random.rand(Float64, (2 * order - 1))
        sort!(xs, rev=true)

        # -- Map unit-interval points to contour times on the imaginary time branch
        τs = teval.timeordered_unit_interval_points_to_imaginary_branch_inch_worm_times(
            contour, worm_nodes, x1, xs)

        # -- Sanity checks for the time points `τs`
        τ_f, τ_w, τ_0 = [ node.time for node in worm_nodes ]

        # First time τs[1] is between the final- and worm-time
        @test τ_w <= τs[1] <= τ_f

        # All other times τs[2:end] are between the worm- and initial-time
        @test all([τ_i <= τ_w for τ_i in τs[2:end]])
        @test all([τ_i >= τ_0 for τ_i in τs[2:end]])

        # -- Evaluate all diagrams at `order`
        value = teval.eval(expansion, worm_nodes, τs, diagrams)
        println("value = $value")

        accumulated_value += value
    end
    println("accumulated_value = $accumulated_value")

end
