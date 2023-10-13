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

"""
High level functions implementing the quasi Monte Carlo inchworm algorithm.

# Exports
$(EXPORTS)
"""
module inchworm

using DocStringExtensions

using TimerOutputs: TimerOutput, @timeit
using ProgressBars: ProgressBar
using Logging

using MPI: MPI
using LinearAlgebra: tr

using Keldysh; kd = Keldysh

using QInchworm.sector_block_matrix: SectorBlockMatrix
using QInchworm.ppgf: partition_function

using QInchworm; teval = QInchworm.topology_eval
using QInchworm.diagrammatics: get_topologies_at_order

using QInchworm.utility: SobolSeqWith0, next!, arbitrary_skip!
using QInchworm.utility: split_count
using QInchworm.mpi: ismaster, rank_sub_range, all_reduce!

using QInchworm.expansion: Expansion, set_bold_ppgf!, set_bold_ppgf_at_order!
using QInchworm.configuration: Configuration,
                               set_initial_node_time!,
                               set_final_node_time!,
                               set_inchworm_node_time!,
                               set_operator_node_time!,
                               sector_block_matrix_from_ppgf
using QInchworm.configuration: Node, InchNode, OperatorNode

using QInchworm.qmc_integrate: qmc_time_ordered_integral_root,
                               qmc_inchworm_integral_root

export inchworm!, correlator_2p

"""
$(TYPEDEF)

Inchworm algorithm input data specific to a given set of topologies.

# Fields
$(TYPEDFIELDS)
"""
struct TopologiesInputData
    "Expansion order"
    order::Int
    "Number of points in the after-``t_w`` region"
    n_pts_after::Int
    "List of contributing topologies"
    topologies::Vector{teval.Topology}
    "Numbers of qMC samples (must be a power of 2)"
    N_samples::Int
end

# http://patorjk.com/software/taag/#p=display&f=Graffiti&t=QInchWorm
const logo = raw"""________  .___              .__    __      __
\_____  \ |   | ____   ____ |  |__/  \    /  \___________  _____
 /  / \  \|   |/    \_/ ___\|  |  \   \/\/   /  _ \_  __ \/     \
/   \_/.  \   |   |  \  \___|   Y  \        (  <_> )  | \/  Y Y  \
\_____\ \_/___|___|  /\___  >___|  /\__/\  / \____/|__|  |__|_|  /
       \__>        \/     \/     \/      \/                    \/ """

#
# Inchworm / bold propagator accumulation functions
#

"""
    $(TYPEDSIGNATURES)

Perform one regular step of qMC inchworm accumulation of the bold propagators.

# Parameters
- `expansion`: Strong coupling expansion problem.
- `c`:         Time contour for integration.
- `τ_i`:       Initial time of the bold propagator to be computed.
- `τ_w`:       Inchworm splitting time ``\\tau_w``.
- `τ_f`:       Final time of the bold propagator to be computed.
- `top_data`:  Inchworm algorithm input data.
- `tmr`:       A `TimerOutput` object used for profiling.

# Returns
Accumulated value of the bold propagator.
"""
function inchworm_step(expansion::Expansion,
                       c::kd.AbstractContour,
                       τ_i::kd.TimeGridPoint,
                       τ_w::kd.TimeGridPoint,
                       τ_f::kd.TimeGridPoint,
                       top_data::Vector{TopologiesInputData};
                       tmr::TimerOutput = TimerOutput())

    t_i, t_w, t_f = τ_i.bpoint, τ_w.bpoint, τ_f.bpoint
    n_i, n_w, n_f = teval.IdentityNode(t_i), teval.InchNode(t_w), teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_w.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    orders = unique(map(td -> td.order, top_data))
    order_contribs = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)

    for td in top_data

        @timeit tmr "Bold" begin
        @timeit tmr "Order $(td.order)" begin
        @timeit tmr "Integration" begin

        order_contrib = deepcopy(zero_sector_block_matrix)

        if td.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_i, 2 => n_w, 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, fixed_nodes, tmr=tmr)
            end # tmr
            @timeit tmr "Evaluation" begin
            order_contrib = eval(td.topologies, kd.BranchPoint[])
            end # tmr
        else
            td.N_samples <= 0 && continue

            @timeit tmr "Setup" begin
            d_after = td.n_pts_after
            d_before = 2 * td.order - td.n_pts_after

            fixed_nodes = Dict(1 => n_i, d_before + 2 => n_w, 2 * td.order + 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, td.order, fixed_nodes, tmr=tmr)

            N_range = rank_sub_range(td.N_samples)
            rank_weight = length(N_range) / td.N_samples

            seq = SobolSeqWith0(2 * td.order)
            arbitrary_skip!(seq, first(N_range) - 1)
            end # tmr

            @timeit tmr "Evaluation" begin
            order_contrib = rank_weight * qmc_inchworm_integral_root(
                t -> eval(td.topologies, t),
                d_before, d_after,
                c, t_i, t_w, t_f,
                init = deepcopy(zero_sector_block_matrix),
                seq = seq,
                N = length(N_range)
            )
            end # tmr

            @timeit tmr "MPI all_reduce" begin
            all_reduce!(order_contrib, +)
            end # tmr
        end

        order_contribs[td.order] += order_contrib

        end; end; end # tmr

    end

    for order in orders
        set_bold_ppgf_at_order!(expansion, order, τ_i, τ_f, order_contribs[order])
    end

    return sum(values(order_contribs))
end

"""
    $(TYPEDSIGNATURES)

Perform the initial step of qMC inchworm accumulation of the bold propagators.
This step amounts to summing all (not only inchworm-proper) diagrams built out of the
bare propagators.

# Parameters
- `expansion`: Strong coupling expansion problem.
- `c`:         Time contour for integration.
- `τ_i`:       Initial time of the bold propagator to be computed.
- `τ_f`:       Final time of the bold propagator to be computed.
- `top_data`:  Inchworm algorithm input data.
- `tmr`:       A `TimerOutput` object used for profiling.

# Returns
Accumulated value of the bold propagator.
"""
function inchworm_step_bare(expansion::Expansion,
                            c::kd.AbstractContour,
                            τ_i::kd.TimeGridPoint,
                            τ_f::kd.TimeGridPoint,
                            top_data::Vector{TopologiesInputData};
                            tmr::TimerOutput = TimerOutput())

    t_i, t_f = τ_i.bpoint, τ_f.bpoint
    n_i, n_f = teval.IdentityNode(t_i), teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)
    result = deepcopy(zero_sector_block_matrix)

    for td in top_data

        @timeit tmr "Bare" begin
        @timeit tmr "Order $(td.order)" begin
        @timeit tmr "Integration" begin

        order_contrib = deepcopy(zero_sector_block_matrix)

        if td.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_i, 2 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, fixed_nodes, tmr=tmr)
            end # tmr
            @timeit tmr "Evaluation" begin
            order_contrib = eval(td.topologies, kd.BranchPoint[])
            end # tmr
        else
            td.N_samples <= 0 && continue

            @timeit tmr "Setup" begin
            d = 2 * td.order

            fixed_nodes = Dict(1 => n_i, 2 * td.order + 2 => n_f)
            eval = teval.TopologyEvaluator(expansion, td.order, fixed_nodes, tmr=tmr)

            N_range = rank_sub_range(td.N_samples)
            rank_weight = length(N_range) / td.N_samples

            seq = SobolSeqWith0(d)
            arbitrary_skip!(seq, first(N_range) - 1)
            end # tmr

            @timeit tmr "Evaluation" begin
            order_contrib = rank_weight * qmc_time_ordered_integral_root(
                t -> eval(td.topologies, t),
                d,
                c, t_i, t_f,
                init = deepcopy(zero_sector_block_matrix),
                seq = seq,
                N = length(N_range)
            )
            end # tmr

            @timeit tmr "MPI all_reduce" begin
            all_reduce!(order_contrib, +)
            end # tmr
        end

        set_bold_ppgf_at_order!(expansion, td.order, τ_i, τ_f, order_contrib)
        result += order_contrib

        end; end; end # tmr

    end
    return result
end

"""
    $(TYPEDSIGNATURES)

Perform a complete qMC inchworm calculation of the bold propagators on the imaginary
time segment. Results of the calculation are written into `expansion.P`.

# Parameters
- `expansion`:       Strong coupling expansion problem.
- `grid`:            Imaginary time grid of the bold propagators.
- `orders`:          List of expansion orders to be accounted for during a regular
                     inchworm step.
- `orders_bare`:     List of expansion orders to be accounted for during the initial
                     inchworm step.
- `N_samples`:       Number of samples to be used in qMC integration. Must be a power of 2.
- `n_pts_after_max`: Maximum number of points in the after-``\\tau_w`` region to be taken
                     into account. By default, diagrams with all valid numbers of the
                     after-``\\tau_w`` points are considered.
"""
function inchworm!(expansion::Expansion,
                   grid::kd.ImaginaryTimeGrid,
                   orders,
                   orders_bare,
                   N_samples::Int64;
                   n_pts_after_max::Int64 = typemax(Int64))

    tmr = TimerOutput()

    @assert N_samples == 0 || ispow2(N_samples)

    if ismaster()
        comm = MPI.COMM_WORLD
        comm_size = MPI.Comm_size(comm)
        N_split = split_count(N_samples, comm_size)

        @info """

        $(logo)

        n_τ = $(length(grid))
        orders_bare = $(orders_bare)
        orders = $(orders)
        n_pts_after_max = $(n_pts_after_max == typemax(Int64) ? "unrestricted" :
                                                                n_pts_after_max)
        # qMC samples = $(N_samples)
        # MPI ranks = $(comm_size)
        # qMC samples (per rank, min:max) = $(minimum(N_split)):$(maximum(N_split))
        """
    end

    # Extend expansion.P_orders to max of orders, orders_bare
    max_order = maximum([maximum(orders), maximum(orders_bare)])
    for o in 1:(max_order+1)
        push!(expansion.P_orders, kd.zero(expansion.P0))
    end

    # First inchworm step

    top_data = TopologiesInputData[]
    for order in orders_bare

        @timeit tmr "Bare" begin
        @timeit tmr "Order $(order)" begin
        @timeit tmr "Topologies" begin

        topologies = get_topologies_at_order(order)
        push!(top_data, TopologiesInputData(order, 2*order, topologies, N_samples))

        end; end; end # tmr "Bare" "Order" "Topologies"

    end

    if ismaster()
        msg = prod(["Bare order $(d.order), # topologies = $(length(d.topologies))\n"
                    for d in top_data])
        @info "Diagrams with bare propagators\n$(msg)"
    end

    if ismaster()
        @info "Initial inchworm step: Evaluating diagrams with bare propagators"
    end

    result = inchworm_step_bare(expansion,
                                grid.contour,
                                grid[1],
                                grid[2],
                                top_data,
                                tmr=tmr)
    set_bold_ppgf!(expansion, grid[1], grid[2], result)

    # The rest of inching

    empty!(top_data)

    for order in orders

        @timeit tmr "Bold" begin
        @timeit tmr "Order $(order)" begin
        @timeit tmr "Topologies" begin

        if order == 0
            n_pts_after_range = 0:0
        else
            n_pts_after_range = 1:min(2 * order - 1, n_pts_after_max)
        end

        for n_pts_after in n_pts_after_range
            topologies = get_topologies_at_order(order, n_pts_after)

            if !isempty(topologies)
                push!(top_data,
                    TopologiesInputData(order, n_pts_after, topologies, N_samples)
                )
            end
        end

        end; end; end # tmr "Bold" "Order" "Topologies"

    end

    if ismaster()
        msg = prod(["Bold order $(d.order), " *
                    "n_pts_after $(d.n_pts_after), " *
                    "# topologies = $(length(d.topologies))\n"
                    for d in top_data])
        @info "Diagrams with bold propagators\n$(msg)"
    end

    if ismaster()
        @info "Evaluating diagrams with bold propagators"
    end

    iter = 2:length(grid)-1
    if ismaster()
        logger = Logging.current_logger()
        if isa(logger, Logging.ConsoleLogger) && logger.min_level <= Logging.Info
            iter = ProgressBar(iter)
        end
    end

    τ_i = grid[1]
    for n in iter
        τ_w = grid[n]
        τ_f = grid[n + 1]

        result = inchworm_step(expansion, grid.contour, τ_i, τ_w, τ_f, top_data, tmr=tmr)
        set_bold_ppgf!(expansion, τ_i, τ_f, result)
    end

    ismaster() && @debug string("Timed sections in inchworm!()\n", tmr)

    return Nothing
end

#
# Calculation of two-point correlators on the Matsubara branch
#

"""
    $(TYPEDSIGNATURES)

Calculate value of a two-point correlator ``\\langle A(\\tau) B(0)\\rangle`` for
one value of the imaginary time argument ``\\tau``. The pair of operators ``(A, B)`` used in
the calculation is taken from `expansion.corr_operators[A_B_pair_idx]`.

# Parameters
- `expansion`:    Strong coupling expansion problem. `expansion.P` must contain precomputed
                  bold propagators.
- `grid`:         Imaginary time grid of the correlator to be computed.
- `A_B_pair_idx`: Index of the ``(A, B)`` pair within `expansion.corr_operators`.
- `τ`:            The imaginary time argument ``\\tau``.
- `top_data`:     Accumulation input data.
- `tmr`:          A `TimerOutput` object used for profiling.

# Returns
Accumulated value of the two-point correlator.
"""
function correlator_2p(expansion::Expansion,
                       grid::kd.ImaginaryTimeGrid,
                       A_B_pair_idx::Int64,
                       τ::kd.TimeGridPoint,
                       top_data::Vector{TopologiesInputData};
                       tmr::TimerOutput = TimerOutput())::ComplexF64
    t_B = grid[1].bpoint # B is always placed at τ=0
    t_A = τ.bpoint
    t_f = grid[end].bpoint

    n_B = teval.OperatorNode(t_B, A_B_pair_idx, 2)
    n_A = teval.OperatorNode(t_A, A_B_pair_idx, 1)
    n_f = teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_A.time.ref >= n_B.time.ref

    result::ComplexF64 = 0

    for td in top_data

        @timeit tmr "Order $(td.order)" begin
        @timeit tmr "Integration" begin

        order_contrib::ComplexF64 = 0

        if td.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_B, 2 => n_A, 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, fixed_nodes, tmr=tmr)
            end # tmr
            @timeit tmr "Evaluation" begin
            order_contrib = tr(eval(td.topologies, kd.BranchPoint[]))
            end # tmr
        else
            td.N_samples <= 0 && continue

            @timeit tmr "Setup" begin
            d_after = td.n_pts_after
            d_before = 2 * td.order - td.n_pts_after

            fixed_nodes = Dict(1 => n_B, d_before + 2 => n_A, 2 * td.order + 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, td.order, fixed_nodes, tmr=tmr)

            N_range = rank_sub_range(td.N_samples)
            rank_weight = length(N_range) / td.N_samples

            seq = SobolSeqWith0(2 * td.order)
            arbitrary_skip!(seq, first(N_range) - 1)
            end # tmr

            @timeit tmr "Evaluation" begin
            order_contrib = rank_weight * qmc_inchworm_integral_root(
                t -> tr(eval(td.topologies, t)),
                d_before, d_after,
                grid.contour, t_B, t_A, t_f,
                init = ComplexF64(0),
                seq = seq,
                N = length(N_range)
            )
            end # tmr

            @timeit tmr "MPI all_reduce" begin
            order_contrib = MPI.Allreduce(order_contrib, +, MPI.COMM_WORLD)
            end # tmr
        end

        result += order_contrib

        end; end # tmr
    end

    return result / partition_function(expansion.P)
end

"""
    $(TYPEDSIGNATURES)

Calculate a two-point correlator ``\\langle A(\\tau) B(0)\\rangle`` on the imaginary
time segment. Accumulation is performed for each pair of operators ``(A, B)`` in
`expansion.corr_operators`. Only the operators that are a single monomial in
``c/c^\\dagger`` are supported.

# Parameters
- `expansion`: Strong coupling expansion problem. `expansion.P` must contain precomputed
               bold propagators.
- `grid`:      Imaginary time grid of the correlator to be computed.
- `orders`:    List of expansion orders to be accounted for.
- `N_samples`: Number of samples to be used in qMC integration. Must be a power of 2.

# Returns
A list of scalar-valued GF objects containing the computed correlators, one element per a
pair in `expansion.corr_operators`.
"""
function correlator_2p(expansion::Expansion,
                       grid::kd.ImaginaryTimeGrid,
                       orders,
                       N_samples::Int64)::Vector{kd.ImaginaryTimeGF{ComplexF64, true}}

    tmr = TimerOutput()

    @assert N_samples == 0 || ispow2(N_samples)
    @assert grid.contour.β == expansion.P[1].grid.contour.β

    if ismaster()
        comm = MPI.COMM_WORLD
        comm_size = MPI.Comm_size(comm)
        N_split = split_count(N_samples, comm_size)

        @info """

        $(logo)

        n_τ = $(length(grid))
        orders = $(orders)
        # qMC samples = $(N_samples)
        # MPI ranks = $(comm_size)
        # qMC samples (per rank, min:max) = $(minimum(N_split)):$(maximum(N_split))
        """
    end

    τ_B = grid[1]

    # Pre-compute topologies: These are common for all
    # pairs of operators in expansion.corr_operators.
    top_data = TopologiesInputData[]
    for order in orders

        @timeit tmr "Order $(order)" begin
        @timeit tmr "Topologies" begin

        n_pts_after_range = (order == 0) ? (0:0) : (1:(2 * order - 1))
        for n_pts_after in n_pts_after_range
            topologies = get_topologies_at_order(order, n_pts_after, with_external_arc=true)

            if !isempty(topologies)
                push!(top_data,
                    TopologiesInputData(order, n_pts_after, topologies, N_samples)
                )
            end
        end

        end; end # tmr "Order" "Topologies"

    end

    if ismaster()
        msg = prod(["Order $(d.order), " *
                    "n_pts_after $(d.n_pts_after), " *
                    "# topologies = $(length(d.topologies))\n"
                    for d in top_data])
        @info "Diagrams\n$(msg)"
    end

    # Accumulate correlators
    corr_list = kd.ImaginaryTimeGF{ComplexF64, true}[]
    for (op_pair_idx, (A, B)) in enumerate(expansion.corr_operators)

        @assert length(A) == 1 "Operator A must be a single monomial in C/C^+"
        @assert length(B) == 1 "Operator B must be a single monomial in C/C^+"

        # Create a GF container to carry the result
        is_fermion(expr) = isodd(length(first(keys(expr.monomials))))
        ξ = (is_fermion(A) && is_fermion(B)) ? kd.fermionic : kd.bosonic

        push!(corr_list, kd.ImaginaryTimeGF(grid, 1, ξ, true))

        #
        # Fill in values
        #

        ismaster() && @info "Evaluating correlator ⟨$(A), $(B)⟩"

        # Only the 0-th order can contribute at τ_A = τ_B
        if top_data[1].order == 0
            corr_list[end][τ_B, τ_B] = correlator_2p(
                expansion,
                grid,
                op_pair_idx,
                τ_B,
                top_data[1:1],
                tmr=tmr
            )
        end

        # The rest of τ_A values
        iter = 2:length(grid)
        if ismaster()
            logger = Logging.current_logger()
            if isa(logger, Logging.ConsoleLogger) && logger.min_level <= Logging.Info
                iter = ProgressBar(iter)
            end
        end

        for n in iter
            τ_A = grid[n]
            corr_list[end][τ_A, τ_B] = correlator_2p(
                expansion,
                grid,
                op_pair_idx,
                τ_A,
                top_data,
                tmr=tmr
            )
        end
    end

    ismaster() && @debug string("Timed sections in correlator_2p()\n", tmr)

    return corr_list
end

end # module inchworm
