# QInchworm.jl
#
# Copyright (C) 2021-2024 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
using LinearAlgebra: tr, diagm

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED

using QInchworm.sector_block_matrix: SectorBlockMatrix
using QInchworm.ppgf: partition_function, set_ppgf!, normalize!

using QInchworm; teval = QInchworm.topology_eval
using QInchworm.diagrammatics: get_topologies_at_order

using QInchworm.scrambled_sobol: ScrambledSobolSeq, next!, skip!
using QInchworm.utility: split_count
using QInchworm.mpi: ismaster, rank_sub_range, all_reduce!

using QInchworm.expansion: Expansion, AllPPGFTypes
using QInchworm.configuration: Configuration,
                               set_initial_node_time!,
                               set_final_node_time!,
                               set_inchworm_node_time!,
                               set_operator_node_time!,
                               sector_block_matrix_from_ppgf
using QInchworm.configuration: Node, InchNode, OperatorNode

using QInchworm.qmc_integrate: contour_integral,
                               RootTransform,
                               DoubleSimplexRootTransform
using QInchworm.randomization: RandomizationParams,
                               RequestStdDev,
                               mean_std_from_randomization

export inchworm!, diff_inchworm!, correlator_2p

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
    "Number of qMC samples per sequence (should be a power of 2)"
    N_samples::Int
    "qMC randomization parameters"
    rand_params::RandomizationParams
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
- `seq_type`:  Type of the (quasi-)random sequence to be used for integration.
- `tmr`:       A `TimerOutput` object used for profiling.

# Returns
- Accumulated value of the bold propagator.
- Order-resolved contributions to the bold propagator as a dictionary
  `Dict{Int, SectorBlockMatrix}`.
- Estimated standard deviations of the order-resolved contributions as a dictionary
  `Dict{Int, SectorBlockMatrix}`.
"""
function inchworm_step(expansion::Expansion,
                       c::kd.AbstractContour,
                       τ_i::kd.TimeGridPoint,
                       τ_w::kd.TimeGridPoint,
                       τ_f::kd.TimeGridPoint,
                       top_data::Vector{TopologiesInputData};
                       seq_type::Type{SeqType} = ScrambledSobolSeq,
                       tmr::TimerOutput = TimerOutput()) where SeqType

    t_i, t_w, t_f = τ_i.bpoint, τ_w.bpoint, τ_f.bpoint
    n_i, n_w, n_f = teval.IdentityNode(t_i), teval.InchNode(t_w), teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_w.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    orders = unique(map(td -> td.order, top_data))
    P_order_contribs = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)
    P_order_contribs_std = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)

    for td in top_data

        @timeit tmr "Bold" begin
        @timeit tmr "Order $(td.order)" begin
        @timeit tmr "Integration" begin

        if td.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_i, 2 => n_w, 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, true, fixed_nodes, tmr=tmr)
            end # tmr
            @timeit tmr "Evaluation" begin
            # This evaluation result is exact, so no need to update P_order_contribs_std
            P_order_contribs[td.order] += eval(td.topologies, kd.BranchPoint[])
            end # tmr
        else
            td.N_samples <= 0 && continue

            @timeit tmr "Setup" begin
            d_after = td.n_pts_after
            d_before = 2 * td.order - td.n_pts_after

            fixed_nodes = Dict(1 => n_i, d_before + 2 => n_w, 2 * td.order + 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, td.order, true, fixed_nodes, tmr=tmr)
            trans = DoubleSimplexRootTransform(d_before, d_after, c, t_i, t_w, t_f)

            N_range = rank_sub_range(td.N_samples)
            rank_weight = length(N_range) / td.N_samples
            end # tmr

            @timeit tmr "Evaluation" begin
            contrib_mean, contrib_std =
            mean_std_from_randomization(2 * td.order,
                                        td.rand_params,
                                        seq_type=seq_type) do seq
                skip!(seq, first(N_range) - 1, exact=true)
                res::SectorBlockMatrix = (rank_weight == 0.0) ?
                deepcopy(zero_sector_block_matrix) :
                rank_weight * contour_integral(
                    t -> eval(td.topologies, t),
                    c,
                    trans,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = length(N_range)
                )
                @timeit tmr "MPI all_reduce" begin
                all_reduce!(res, +)
                end # tmr
                res
            end
            P_order_contribs[td.order] += contrib_mean
            P_order_contribs_std[td.order] += contrib_std
            end # tmr
        end

        end; end; end # tmr

    end

    return sum(values(P_order_contribs)), P_order_contribs, P_order_contribs_std
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
- `seq_type`:  Type of the (quasi-)random sequence to be used for integration.
- `tmr`:       A `TimerOutput` object used for profiling.

# Returns
- Accumulated value of the bold propagator.
- Order-resolved contributions to the bold propagator as a dictionary
  `Dict{Int, SectorBlockMatrix}`.
- Estimated standard deviations of the order-resolved contributions as a dictionary
  `Dict{Int, SectorBlockMatrix}`.
"""
function inchworm_step_bare(expansion::Expansion,
                            c::kd.AbstractContour,
                            τ_i::kd.TimeGridPoint,
                            τ_f::kd.TimeGridPoint,
                            top_data::Vector{TopologiesInputData};
                            seq_type::Type{SeqType} = ScrambledSobolSeq,
                            tmr::TimerOutput = TimerOutput()) where SeqType

    t_i, t_f = τ_i.bpoint, τ_f.bpoint
    n_i, n_f = teval.IdentityNode(t_i), teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    orders = unique(map(td -> td.order, top_data))
    P_order_contribs = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)
    P_order_contribs_std = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)

    for td in top_data

        @timeit tmr "Bare" begin
        @timeit tmr "Order $(td.order)" begin
        @timeit tmr "Integration" begin

        if td.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_i, 2 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, false, fixed_nodes, tmr=tmr)
            end # tmr
            @timeit tmr "Evaluation" begin
            P_order_contribs[td.order] = eval(td.topologies, kd.BranchPoint[])
            # This evaluation result is exact, so no need to update P_order_contribs_std
            end # tmr
        else
            td.N_samples <= 0 && continue

            @timeit tmr "Setup" begin
            d = 2 * td.order

            fixed_nodes = Dict(1 => n_i, 2 * td.order + 2 => n_f)
            eval = teval.TopologyEvaluator(expansion, td.order, false, fixed_nodes, tmr=tmr)
            trans = RootTransform(d, c, t_i, t_f)

            N_range = rank_sub_range(td.N_samples)
            rank_weight = length(N_range) / td.N_samples
            end # tmr

            @timeit tmr "Evaluation" begin
            contrib_mean, contrib_std =
            mean_std_from_randomization(d, td.rand_params, seq_type=seq_type) do seq
                skip!(seq, first(N_range) - 1, exact=true)
                res::SectorBlockMatrix = (rank_weight == 0.0) ?
                deepcopy(zero_sector_block_matrix) :
                rank_weight * contour_integral(
                    t -> eval(td.topologies, t),
                    c,
                    trans,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = length(N_range)
                )
                @timeit tmr "MPI all_reduce" begin
                all_reduce!(res, +)
                end # tmr
                res
            end
            P_order_contribs[td.order] += contrib_mean
            P_order_contribs_std[td.order] += contrib_std
            end # tmr
        end

        end; end; end # tmr

    end

    return sum(values(P_order_contribs)), P_order_contribs, P_order_contribs_std
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
- `n_pts_after_max`: Maximum number of points in the after-``\\tau_w`` region to be
                     taken into account. By default, diagrams with all valid numbers of
                     the after-``\\tau_w`` points are considered.
- `rand_params`:     Parameters of the randomized qMC integration.
- `seq_type`:        Type of the (quasi-)random sequence to be used for integration.

# Returns
- Order-resolved contributions to the bold propagator as a dictionary
  `Dict{Int, PPGF}`.
- Estimated standard deviations of the order-resolved contributions as a dictionary
  `Dict{Int, PPGF}`.
"""
function inchworm!(expansion::Expansion,
                   grid::kd.ImaginaryTimeGrid,
                   orders,
                   orders_bare,
                   N_samples::Int64;
                   n_pts_after_max::Int64 = typemax(Int64),
                   rand_params::RandomizationParams = RandomizationParams(),
                   seq_type::Type{SeqType} = ScrambledSobolSeq) where SeqType

    tmr = TimerOutput()

    @assert N_samples == 0 || ispow2(N_samples)
    @assert rand_params.N_seqs > 0

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
        $(rand_params)
        """
    end

    # Prepare containers for order-resolved contributions to the bold propagator
    orders_all = union(orders, orders_bare)
    P_orders = Dict(order => kd.zero(expansion.P) for order in orders_all)
    P_orders_std = Dict(order => kd.zero(expansion.P) for order in orders_all)

    # First inchworm step
    top_data = TopologiesInputData[]
    for order in orders_bare

        @timeit tmr "Bare" begin
        @timeit tmr "Order $(order)" begin
        @timeit tmr "Topologies" begin

        topologies = get_topologies_at_order(order)

        push!(top_data,
              TopologiesInputData(order,
                                  2 * order,
                                  topologies,
                                  N_samples,
                                  rand_params)
        )

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

    result, P_order_contribs, P_order_contribs_std =
        inchworm_step_bare(expansion,
                           grid.contour,
                           grid[1], grid[2],
                           top_data,
                           seq_type=seq_type,
                           tmr=tmr)

    set_ppgf!(expansion.P, grid[1], grid[2], result)
    for order in keys(P_order_contribs)
        set_ppgf!(P_orders[order], grid[1], grid[2], P_order_contribs[order])
        set_ppgf!(P_orders_std[order], grid[1], grid[2], P_order_contribs_std[order])
    end

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
                      TopologiesInputData(
                          order,
                          n_pts_after,
                          topologies,
                          N_samples,
                          rand_params
                      )
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

        result, P_order_contribs, P_order_contribs_std =
            inchworm_step(expansion,
                          grid.contour,
                          τ_i, τ_w, τ_f,
                          top_data,
                          seq_type=seq_type,
                          tmr=tmr)

        set_ppgf!(expansion.P, τ_i, τ_f, result)
        normalize!(expansion.P, τ_f) # Suppress exponential growth by normalization
        for order in keys(P_order_contribs)
            set_ppgf!(P_orders[order], τ_i, τ_f, P_order_contribs[order])
            set_ppgf!(P_orders_std[order], τ_i, τ_f, P_order_contribs_std[order])
        end
    end

    ismaster() && @debug string("Timed sections in inchworm!()\n", tmr)

    return P_orders, P_orders_std
end

#
# Differential inchworm / bold propagator accumulation functions
#

"""
    $(TYPEDSIGNATURES)

Perform one step of the differential qMC inchworm accumulation of the bold propagators.

# Parameters
- `expansion`:   Strong coupling expansion problem.
- `c`:           Imaginary time contour for integration.
- `τ_i`:         Initial time of the bold propagator to be computed.
- `τ_f_prev`:    Final time of the bold propagator computed at the previous step.
- `τ_f`:         Final time of the bold propagator to be computed.
- `Σ`:           Container to store the pseudo-particle self-energy.
- `hamiltonian`: Atomic Hamiltonian.
- `top_data`:    Inchworm algorithm input data.
- `seq_type`:    Type of the (quasi-)random sequence to be used for integration.
- `tmr`:         A `TimerOutput` object used for profiling.

# Returns
- Accumulated value of the bold propagator.
- Order-resolved contributions to pseudo-particle self-energy as a dictionary
  `Dict{Int, SectorBlockMatrix}`.
- Estimated standard deviations of the order-resolved contributions as a dictionary
  `Dict{Int, SectorBlockMatrix}`.
"""
function diff_inchworm_step!(expansion::Expansion,
                             c::kd.ImaginaryContour,
                             τ_i::kd.TimeGridPoint,
                             τ_f_prev::kd.TimeGridPoint,
                             τ_f::kd.TimeGridPoint,
                             Σ::PPGF,
                             hamiltonian::SectorBlockMatrix,
                             top_data::Vector{TopologiesInputData};
                             seq_type::Type{SeqType} = ScrambledSobolSeq,
                             tmr::TimerOutput = TimerOutput()
                             ) where {PPGF <: AllPPGFTypes, SeqType}

    t_i, t_f_prev, t_f = τ_i.bpoint, τ_f_prev.bpoint, τ_f.bpoint
    @assert t_f.ref >= t_f_prev.ref >= t_i.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    orders = unique(map(td -> td.order, top_data))
    Σ_order_contribs = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)
    Σ_order_contribs_std = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)

    # Value of the bold propagator at the previous time slice
    P_prev = Dict(s => (s, p[τ_f_prev, τ_i]) for (s, p) in enumerate(expansion.P))

    for td in top_data

        @assert td.order > 0 "Pseudo-particle self-energy has no 0-th order contribution"

        @timeit tmr "Order $(td.order)" begin
        @timeit tmr "Integration" begin

        if td.order == 1
            @timeit tmr "Evaluation" begin
            fixed_nodes = Dict(1 => teval.PairNode(t_i), 2 => teval.PairNode(t_f_prev))
            eval = teval.TopologyEvaluator(expansion, 1, true, fixed_nodes, tmr=tmr)
            Σ_order_contribs[td.order] = eval(td.topologies, BranchPoint[])
            end # tmr
        else

            td.N_samples <= 0 && continue

            @timeit tmr "Setup" begin
            d = 2 * td.order - 2

            fixed_nodes = Dict(1 => teval.PairNode(t_i), d + 2 => teval.PairNode(t_f_prev))
            eval = teval.TopologyEvaluator(expansion, td.order, true, fixed_nodes, tmr=tmr)
            trans = RootTransform(d, c, t_i, t_f_prev)

            N_range = rank_sub_range(td.N_samples)
            rank_weight = length(N_range) / td.N_samples
            end # tmr

            @timeit tmr "Evaluation" begin
            contrib_mean, contrib_std =
            mean_std_from_randomization(2 * td.order,
                                        td.rand_params,
                                        seq_type=seq_type) do seq
                skip!(seq, first(N_range) - 1, exact=true)
                res::SectorBlockMatrix = (rank_weight == 0.0) ?
                deepcopy(zero_sector_block_matrix) :
                rank_weight * contour_integral(
                    t -> eval(td.topologies, t),
                    c,
                    trans,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = length(N_range)
                )
                @timeit tmr "MPI all_reduce" begin
                all_reduce!(res, +)
                end # tmr
                res
            end
            Σ_order_contribs[td.order] += contrib_mean
            Σ_order_contribs_std[td.order] += contrib_std
            end # tmr
        end

        end; end # tmr

    end

    grid = first(expansion.P).grid

    # Update Σ at time τ_f_prev
    Σ_prev = sum(values(Σ_order_contribs))
    set_ppgf!(Σ, τ_i, τ_f_prev, Σ_prev)

    # Convolution Σ * P
    rhs = SectorBlockMatrix()
    for s in eachindex(expansion.P)
        size = kd.norbitals(expansion.P[s])
        # Keldysh.integrate() uses the trapezoid rule
        conv = kd.integrate(τ -> Σ[s][τ_f_prev, τ] * expansion.P[s][τ, τ_i],
                            grid,
                            τ_f_prev,
                            τ_i,
                            zeros(ComplexF64, size, size))
        rhs[s] = (s, conv)
    end

    # Term governing PPGF evolution in absence of interaction
    rhs += -hamiltonian * P_prev

    # Solve the ODE for the bold propagator using the simple Euler method
    Δτ = 1im * step(grid, kd.imaginary_branch)
    P = P_prev + Δτ * rhs
    return P, Σ_order_contribs, Σ_order_contribs_std
end

"""
    $(TYPEDSIGNATURES)

Perform a complete qMC inchworm calculation of the bold propagators on the imaginary
time segment using the differential formulation of the method described in

```
"Inchworm Monte Carlo Method for Open Quantum Systems"
Z. Cai, J. Lu and S. Yang
Comm. Pure Appl. Math., 73: 2430-2472 (2020)
```

Results of the calculation are written into `expansion.P`.

# Parameters
- `expansion`:   Strong coupling expansion problem.
- `grid`:        Imaginary time grid of the bold propagators.
- `orders`:      List of expansion orders to be accounted for.
- `N_samples`:   Number of samples to be used in qMC integration. Must be a power of 2.
- `rand_params`: Parameters of the randomized qMC integration.
- `seq_type`:    Type of the (quasi-)random sequence to be used for integration.

# Returns
- Order-resolved contributions to the pseudo-particle self-energy as a dictionary
  `Dict{Int, PPGF}`.
- Estimated standard deviations of the order-resolved contributions as a dictionary
  `Dict{Int, PPGF}`.
"""
function diff_inchworm!(expansion::Expansion,
                        grid::kd.ImaginaryTimeGrid,
                        orders,
                        N_samples::Int64;
                        rand_params::RandomizationParams = RandomizationParams(),
                        seq_type::Type{SeqType} = ScrambledSobolSeq
                        ) where SeqType

    tmr = TimerOutput()

    @assert N_samples == 0 || ispow2(N_samples)
    @assert rand_params.N_seqs > 0

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
        $(rand_params)
        """
    end

    # Prepare containers for order-resolved contributions to the self-energy
    Σ_orders = Dict(order => kd.zero(expansion.P) for order in orders)
    Σ_orders_std = Dict(order => kd.zero(expansion.P) for order in orders)

    # Differential inching
    top_data = TopologiesInputData[]
    for order in orders

        # There is no 0-th order contribution to the pseudo-particle self-energy
        order == 0 && continue

        @timeit tmr "Order $(order)" begin
        @timeit tmr "Topologies" begin

        topologies = get_topologies_at_order(order, 1)

        if !isempty(topologies)
            push!(top_data,
                  TopologiesInputData(order, 1, topologies, N_samples, rand_params)
            )
        end

        end; end # tmr "Order" "Topologies"

    end

    if ismaster()
        msg = prod(["Order $(d.order), " *
                    "# topologies = $(length(d.topologies))\n"
                    for d in top_data])
        @info "Diagrams\n$(msg)"
    end

    if ismaster()
        @info "Evaluating diagrams"
    end

    iter = 1:length(grid)-1
    if ismaster()
        logger = Logging.current_logger()
        if isa(logger, Logging.ConsoleLogger) && logger.min_level <= Logging.Info
            iter = ProgressBar(iter)
        end
    end

    # Prepare the Hamiltonian block matrix
    β = first(expansion.P).grid.contour.β
    Z = ked.partition_function(expansion.ed, β)
    λ = log(Z) / β
    hamiltonian = Dict(s => (s, diagm(convert.(ComplexF64, eig.eigenvalues .+ λ)))
                       for (s, eig) in enumerate(expansion.ed.eigensystems))

    # Prepare self-energy container
    Σ = zero(expansion.P)

    τ_i = grid[1]
    for n in iter
        τ_f_prev = grid[n]
        τ_f = grid[n + 1]

        result, Σ_order_contribs, Σ_order_contribs_std =
            diff_inchworm_step!(expansion,
                                grid.contour,
                                τ_i,
                                τ_f_prev,
                                τ_f,
                                Σ,
                                hamiltonian,
                                top_data,
                                tmr=tmr)

        set_ppgf!(expansion.P, τ_i, τ_f, result)
        for order in keys(Σ_order_contribs)
            set_ppgf!(Σ_orders[order], τ_i, τ_f, Σ_order_contribs[order])
            set_ppgf!(Σ_orders_std[order], τ_i, τ_f, Σ_order_contribs_std[order])
        end
    end

    ismaster() && @debug string("Timed sections in diff_inchworm!()\n", tmr)

    return Σ_orders, Σ_orders_std
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
- `seq_type`:     Type of the (quasi-)random sequence to be used for integration.
- `tmr`:          A `TimerOutput` object used for profiling.

# Returns
- Accumulated value of the two-point correlator.
- Estimated standard deviations of the computed correlator.
"""
function correlator_2p(expansion::Expansion,
                       grid::kd.ImaginaryTimeGrid,
                       A_B_pair_idx::Int64,
                       τ::kd.TimeGridPoint,
                       top_data::Vector{TopologiesInputData};
                       seq_type::Type{SeqType} = ScrambledSobolSeq,
                       tmr::TimerOutput = TimerOutput()
                       )::Tuple{ComplexF64, ComplexF64} where SeqType
    t_B = grid[1].bpoint # B is always placed at τ=0
    t_A = τ.bpoint
    t_f = grid[end].bpoint

    n_B = teval.OperatorNode(t_B, A_B_pair_idx, 2)
    n_A = teval.OperatorNode(t_A, A_B_pair_idx, 1)
    n_f = teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_A.time.ref >= n_B.time.ref

    result::ComplexF64 = 0
    result_std::Float64 = 0

    for td in top_data

        @timeit tmr "Order $(td.order)" begin
        @timeit tmr "Integration" begin

        order_contrib::ComplexF64 = 0
        order_contrib_std::Float64 = 0

        if td.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_B, 2 => n_A, 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, true, fixed_nodes, tmr=tmr)
            end # tmr
            @timeit tmr "Evaluation" begin
            order_contrib = tr(eval(td.topologies, kd.BranchPoint[]))
            order_contrib_std = .0
            end # tmr
        else
            td.N_samples <= 0 && continue

            @timeit tmr "Setup" begin
            d_after = td.n_pts_after
            d_before = 2 * td.order - td.n_pts_after

            fixed_nodes = Dict(1 => n_B, d_before + 2 => n_A, 2 * td.order + 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, td.order, true, fixed_nodes, tmr=tmr)
            trans = DoubleSimplexRootTransform(d_before,
                                               d_after,
                                               grid.contour,
                                               t_B, t_A, t_f)

            N_range = rank_sub_range(td.N_samples)
            rank_weight = length(N_range) / td.N_samples
            end # tmr

            @timeit tmr "Evaluation" begin
            order_contrib, order_contrib_std =
            mean_std_from_randomization((2 * td.order),
                                        td.rand_params,
                                        seq_type=seq_type) do seq
                skip!(seq, first(N_range) - 1, exact=true)
                res::ComplexF64 = (rank_weight == 0.0) ?
                deepcopy(zero_sector_block_matrix) :
                rank_weight * contour_integral(
                    t -> tr(eval(td.topologies, t)),
                    grid.contour,
                    trans,
                    init = ComplexF64(0),
                    seq = seq,
                    N = length(N_range)
                )
                @timeit tmr "MPI all_reduce" begin
                MPI.Allreduce(res, +, MPI.COMM_WORLD)
                end # tmr
            end
            end # tmr
        end

        result += order_contrib
        result_std += order_contrib_std

        end; end # tmr
    end

    return (result, result_std) ./ partition_function(expansion.P)
end

"""
    $(TYPEDSIGNATURES)

Calculate a two-point correlator ``\\langle A(\\tau) B(0)\\rangle`` on the imaginary
time segment. Accumulation is performed for each pair of operators ``(A, B)`` in
`expansion.corr_operators`. Only the operators that are a single monomial in
``c/c^\\dagger`` are supported.

This method is selected by the flag argument of type [`RequestStdDev`](@ref) and returns
randomized qMC estimates of both mean and standard deviation of the correlators.

# Parameters
- `expansion`:   Strong coupling expansion problem. `expansion.P` must contain precomputed
                 bold propagators.
- `grid`:        Imaginary time grid of the correlator to be computed.
- `orders`:      List of expansion orders to be accounted for.
- `N_samples`:   Number of samples to be used in qMC integration. Must be a power of 2.
- `rand_params`: Parameters of the randomized qMC integration.
- `seq_type`:    Type of the (quasi-)random sequence to be used for integration.

# Returns
- A list of scalar-valued GF objects containing the computed correlators, one element per
  a pair in `expansion.corr_operators`.
- A list of scalar-valued GF objects containing estimated standard deviations of
  the computed correlators, one element per a pair in `expansion.corr_operators`.
"""
function correlator_2p(expansion::Expansion,
                       grid::kd.ImaginaryTimeGrid,
                       orders,
                       N_samples::Int64,
                       ::RequestStdDev;
                       rand_params::RandomizationParams = RandomizationParams(),
                       seq_type::Type{SeqType} = ScrambledSobolSeq
                       )::Tuple{Vector{kd.ImaginaryTimeGF{ComplexF64, true}},
                                Vector{kd.ImaginaryTimeGF{ComplexF64, true}}} where SeqType

    tmr = TimerOutput()

    @assert N_samples == 0 || ispow2(N_samples)
    @assert rand_params.N_seqs > 0
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
        $(rand_params)
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
                      TopologiesInputData(
                            order,
                            n_pts_after,
                            topologies,
                            N_samples,
                            rand_params
                        )
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
    corr_std_list = kd.ImaginaryTimeGF{ComplexF64, true}[]

    for (op_pair_idx, (A, B)) in enumerate(expansion.corr_operators)

        @assert length(A) == 1 "Operator A must be a single monomial in C/C^+"
        @assert length(B) == 1 "Operator B must be a single monomial in C/C^+"

        # Create a GF container to carry the result
        is_fermion(expr) = isodd(length(first(keys(expr.monomials))))
        ξ = (is_fermion(A) && is_fermion(B)) ? kd.fermionic : kd.bosonic

        push!(corr_list, kd.ImaginaryTimeGF(grid, 1, ξ, true))
        push!(corr_std_list, kd.ImaginaryTimeGF(grid, 1, ξ, true))

        #
        # Fill in values
        #

        ismaster() && @info "Evaluating correlator ⟨$(A), $(B)⟩"

        # Only the 0-th order can contribute at τ_A = τ_B
        if top_data[1].order == 0
            corr_list[end][τ_B, τ_B], corr_std_list[end][τ_B, τ_B] =
            correlator_2p(
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
            corr_list[end][τ_A, τ_B], corr_std_list[end][τ_A, τ_B] =
            correlator_2p(
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

    return (corr_list, corr_std_list)
end

"""
    $(TYPEDSIGNATURES)

Calculate a two-point correlator ``\\langle A(\\tau) B(0)\\rangle`` on the imaginary
time segment. Accumulation is performed for each pair of operators ``(A, B)`` in
`expansion.corr_operators`. Only the operators that are a single monomial in
``c/c^\\dagger`` are supported.

# Parameters
- `expansion`:   Strong coupling expansion problem. `expansion.P` must contain precomputed
                 bold propagators.
- `grid`:        Imaginary time grid of the correlator to be computed.
- `orders`:      List of expansion orders to be accounted for.
- `N_samples`:   Number of samples to be used in qMC integration. Must be a power of 2.
- `rand_params`: Parameters of the randomized qMC integration.
- `seq_type`:    Type of the (quasi-)random sequence to be used for integration.

# Returns
A list of scalar-valued GF objects containing the computed correlators, one element per
a pair in `expansion.corr_operators`.
"""
function correlator_2p(expansion::Expansion,
    grid::kd.ImaginaryTimeGrid,
    orders,
    N_samples::Int64;
    rand_params::RandomizationParams = RandomizationParams(),
    seq_type::Type{SeqType} = ScrambledSobolSeq
    )::Vector{kd.ImaginaryTimeGF{ComplexF64, true}} where SeqType
    return correlator_2p(expansion, grid, orders, N_samples, RequestStdDev();
                         rand_params=rand_params)[1]
end

end # module inchworm
