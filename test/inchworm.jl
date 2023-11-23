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
# Authors: Igor Krivenko, Hugo U. R. Strand

using Test

using Random: MersenneTwister
using MPI; MPI.Init()
using HDF5

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.spline_gf: SplineInterpolatedGF

using QInchworm.diagrammatics: get_topologies_at_order
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.randomization: RandomizationParams

using QInchworm.inchworm: TopologiesInputData,
                          inchworm_step,
                          inchworm_step_bare,
                          inchworm!,
                          correlator_2p

# If true, write calculation results into inchworm.h5
const write_h5 = false

function test_or_write(fid, name, value)
    if write_h5
        HDF5.write(fid, name, value)
    else
        @test value ≈ HDF5.read(fid, name)
    end
end

# Single state pseudo particle expansion

β = 10.
μ = +0.1 # Chemical potential
ϵ = +0.1 # Bath energy level
V = -0.1 # Hybridization

tmax = 1.
nt = 10
nτ = 20

H = - μ * op.n("0")
soi = ked.Hilbert.SetOfIndices([["0"]])
ed = ked.EDCore(H, soi)
ρ = ked.density_matrix(ed, β)

@testset "inchworm_step" begin
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β))
    grid = kd.FullTimeGrid(contour, nt, nτ)

    # Hybridization propagator

    Δ = V^2 * kd.FullTimeGF(kd.DeltaDOS(ϵ), grid)
    Δ_rev = kd.FullTimeGF((t1, t2) -> -Δ[t2, t1, false], grid, 1, kd.fermionic, true)

    # Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ_rev)
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

    # Extend expansion.P_orders and expansion.P_orders_std to max of orders
    for o in 1:(maximum(orders)+1)
        push!(expansion.P_orders, kd.zero(expansion.P0))
        push!(expansion.P_orders_std, kd.zero(expansion.P0))
    end

    rand_params = RandomizationParams()

    order_data = TopologiesInputData[]
    for order in 0:3
        n_pts_after_range = (order == 0) ? (0:0) : (1:(2 * order - 1))
        for n_pts_after in n_pts_after_range
            topologies = get_topologies_at_order(order, n_pts_after)
            if !isempty(topologies)
                push!(order_data,
                      TopologiesInputData(order,
                                          n_pts_after,
                                          topologies,
                                          N_samples,
                                          rand_params)
                )
            end
        end
    end

    value = inchworm_step(expansion, contour, τ_i, τ_w, τ_f, order_data)

    HDF5.h5open((@__DIR__) * "/inchworm.h5", write_h5 ? "cw" : "r") do fid
        for s = 1:2
            test_or_write(fid, "/inchworm_step/value/$(s)", value[s][2])
        end
    end
end

@testset "inchworm_step_bare" begin
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
    grid = kd.FullTimeGrid(contour, nt, nτ);

    # Hybridization propagator

    Δ = V^2 * kd.FullTimeGF(kd.DeltaDOS(ϵ), grid)
    Δ_rev = kd.FullTimeGF((t1, t2) -> -Δ[t2, t1, false], grid, 1, kd.fermionic, true)

    # Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ_rev)
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
    for o in 1:(maximum(orders)+1)
        push!(expansion.P_orders, kd.zero(expansion.P0))
        push!(expansion.P_orders_std, kd.zero(expansion.P0))
    end

    rand_params = RandomizationParams()

    order_data = TopologiesInputData[]
    for order in 0:3
        topologies = get_topologies_at_order(order)
        push!(order_data, TopologiesInputData(order,
                                              1,
                                              topologies,
                                              N_samples,
                                              rand_params))
    end

    value = inchworm_step_bare(expansion, contour, τ_i, τ_f, order_data)

    HDF5.h5open((@__DIR__) * "/inchworm.h5", write_h5 ? "cw" : "r") do fid
        for s = 1:2
            test_or_write(fid, "/inchworm_step_bare/value/$(s)", value[s][2])
        end
    end
end

@testset "inchworm" begin
    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, nτ);

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ), grid)
    Δ_rev = kd.ImaginaryTimeGF((t1, t2) -> -Δ[t2, t1, false], grid, 1, kd.fermionic, true)

    # Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), SplineInterpolatedGF(Δ))
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), SplineInterpolatedGF(Δ_rev))
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf = true)

    orders = 0:3
    orders_bare = 0:2
    N_samples = 2^8

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    # Single-particle GF

    add_corr_operators!(expansion, (op.c("0"), op.c_dag("0")))
    g = -correlator_2p(expansion, grid, orders, N_samples)

    HDF5.h5open((@__DIR__) * "/inchworm.h5", write_h5 ? "cw" : "r") do fid
        for s = 1:2
            test_or_write(fid, "/inchworm/P/$(s)", expansion.P[s].GF.mat.data)
        end
        test_or_write(fid, "/inchworm/g", g[1].mat.data)
    end
end
