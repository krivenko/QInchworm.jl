using Test
using Random

using MPI; MPI.Init()
using HDF5

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm; cfg = QInchworm.configuration; teval = QInchworm.topology_eval
using QInchworm: SectorBlockMatrix

using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!

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

    Δ = V^2 * kd.FullTimeGF(kd.DeltaDOS(ϵ), grid)
    Δ_rev = kd.FullTimeGF((t1, t2) -> Δ[t2, t1], grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ_rev)
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    # -- Inch-worm node configuration, fixing the final-time and worm-time

    τ_grid = grid[kd.imaginary_branch]

    fidx = 8
    widx = fidx - 1

    n_0 = teval.IdentityNode(τ_grid[1].bpoint)
    n_w = teval.InchNode(τ_grid[widx].bpoint)
    n_f = teval.IdentityNode(τ_grid[fidx].bpoint)

    worm_nodes = [n_0, n_w, n_f]

    # -- Generate all topologies and diagrams at `order`

    order = 3
    topologies = teval.get_topologies_at_order(order, 1)

    println("n_topologies = $(length(topologies))")
    for topology in topologies
        println(topology)
    end

    n_samples = 100

    values = Matrix{ComplexF64}(undef, n_samples, 2)
    accumulated_value = zeros(SectorBlockMatrix, expansion.ed)

    #Random.seed!(1234)
    #x1_list = Random.rand(Float64, n_samples)
    #xs_list = Random.rand(Float64, (2 * order - 1), n_samples)
    #sort!(xs_list, dims=1, rev=true)

    fid = HDF5.h5open((@__DIR__) * "/topology_eval.h5", "r")
    x1_list = HDF5.read(fid["/x1_list"])
    xs_list = HDF5.read(fid["/xs_list"])
    HDF5.close(fid)

    tev = teval.TopologyEvaluator(expansion, order, Dict(1 => n_0, 7 => n_w, 9 => n_f))

    for sample in 1:n_samples

        # -- Generate time ordered points on the unit-interval (replace with quasi-MC points)
        # -- separating the initial point `x1` (between the final- and worm-time) from the others `xs`

        x1 = x1_list[sample]
        xs = xs_list[:, sample]

        # -- Map unit-interval points to contour times on the imaginary time branch
        x_0, x_w, x_f = [ node.time.ref for node in worm_nodes ]

        x1 = x_w .+ (x_f - x_w) * x1
        xs = x_0 .+ (x_w - x_0) * xs
        xs = vcat([x1], xs)

        τs = [ kd.get_point(contour[kd.imaginary_branch], x) for x in xs ]

        # -- Sanity checks for the time points `τs`
        τ_0, τ_w, τ_f = [ node.time for node in worm_nodes ]

        # First time τs[1] is between the final- and worm-time
        @test τ_w <= τs[1] <= τ_f

        # All other times τs[2:end] are between the worm- and initial-time
        @test all([τ_i <= τ_w for τ_i in τs[2:end]])
        @test all([τ_i >= τ_0 for τ_i in τs[2:end]])

        # -- Evaluate all diagrams at `order`
        value = tev(topologies, τs)
        println("value = $value")
        values[sample, :] = [value[1][2][1, 1], value[2][2][1, 1]]

        accumulated_value += value
    end
    println("accumulated_value = $accumulated_value")

    #HDF5.h5open((@__DIR__) * "/topology_eval.h5", "w") do fid
    #    HDF5.write(fid, "/x1_list", x1_list)
    #    HDF5.write(fid, "/xs_list", xs_list)
    #    HDF5.write(fid, "/values", values)
    #end

    HDF5.h5open((@__DIR__) * "/topology_eval.h5", "r") do fid
        @test isapprox(values, HDF5.read(fid["/values"]), rtol=1e-10)
    end

end
