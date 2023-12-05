# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
#
# QInchworm.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# QInchworm.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Hugo U. R. Strand, Igor Krivenko

using Test
using Random

using HDF5

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.diagrammatics: Topology, get_topologies_at_order
using QInchworm.topology_eval: TopologyEvaluator, IdentityNode, InchNode
using QInchworm.sector_block_matrix: SectorBlockMatrix

using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!

# If true, write input data and calculation results into topology_eval.h5
const write_h5 = false

@testset "topology_to_config" begin

    # -- Single state pseudo particle expansion

    β = 10.
    μ = +0.1 # Chemical potential
    ϵ = +0.1 # Bath energy level
    V = -0.1 # Hybridization

    tmax = 1.
    nt = 10
    nτ = 30

    H = - μ * op.n("0")
    soi = ked.Hilbert.SetOfIndices([["0"]])
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
    grid = kd.FullTimeGrid(contour, nt, nτ);

    # -- Hybridization propagator

    Δ = V^2 * kd.FullTimeGF(kd.DeltaDOS(ϵ), grid)
    Δ_rev = kd.FullTimeGF((t1, t2) -> Δ[t2, t1], grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ_rev)
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    # -- Inchworm node configuration, fixing the final-time and worm-time

    τ_grid = grid[kd.imaginary_branch]

    fidx = 8
    widx = fidx - 1

    n_0 = IdentityNode(τ_grid[1].bpoint)
    n_w = InchNode(τ_grid[widx].bpoint)
    n_f = IdentityNode(τ_grid[fidx].bpoint)

    worm_nodes = [n_0, n_w, n_f]

    # -- Generate all topologies and diagrams at `order`

    order = 3
    topologies = get_topologies_at_order(order, 1)

    @test length(topologies) == 4

    n_samples = 100

    values = Matrix{ComplexF64}(undef, n_samples, 2)
    accumulated_value = zeros(SectorBlockMatrix, expansion.ed)

    if write_h5
        Random.seed!(1234)
        x1_list = Random.rand(Float64, n_samples)
        xs_list = Random.rand(Float64, (2 * order - 1), n_samples)
        sort!(xs_list, dims=1, rev=true)
    else
        HDF5.h5open((@__DIR__) * "/topology_eval.h5", "r") do fid
            x1_list = HDF5.read(fid["/x1_list"])
            xs_list = HDF5.read(fid["/xs_list"])
        end
    end

    tev = TopologyEvaluator(expansion, order, Dict(1 => n_0, 7 => n_w, 9 => n_f))

    for sample in 1:n_samples

        # Generate time ordered points on the unit-interval (replace with quasi-MC points)
        # separating the initial point `x1` (between the final- and worm-time) from the
        # others `xs`.

        x1 = x1_list[sample]
        xs = xs_list[:, sample]

        # Map unit-interval points to contour times on the imaginary time branch
        x_0, x_w, x_f = [ node.time.ref for node in worm_nodes ]

        x1 = x_w .+ (x_f - x_w) * x1
        xs = x_0 .+ (x_w - x_0) * xs
        xs = vcat([x1], xs)

        τs = [ kd.get_point(contour[kd.imaginary_branch], x) for x in xs ]

        # Sanity checks for the time points `τs`
        τ_0, τ_w, τ_f = [ node.time for node in worm_nodes ]

        # First time τs[1] is between the final- and worm-time
        @test τ_w <= τs[1] <= τ_f

        # All other times τs[2:end] are between the worm- and initial-time
        @test all([τ_i <= τ_w for τ_i in τs[2:end]])
        @test all([τ_i >= τ_0 for τ_i in τs[2:end]])

        # Evaluate all diagrams at `order`
        value = tev(topologies, τs)
        values[sample, :] = [value[1][2][1, 1], value[2][2][1, 1]]

        accumulated_value += value
    end

    if write_h5
        HDF5.h5open((@__DIR__) * "/topology_eval.h5", "w") do fid
            HDF5.write(fid, "/x1_list", x1_list)
            HDF5.write(fid, "/xs_list", xs_list)
            HDF5.write(fid, "/values", values)
        end
    else
        HDF5.h5open((@__DIR__) * "/topology_eval.h5", "r") do fid
            @test isapprox(values, HDF5.read(fid["/values"]), rtol=1e-10)
        end
    end

end
