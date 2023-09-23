module inchworm

using DocStringExtensions

using TimerOutputs: TimerOutput, @timeit
using ProgressBars: ProgressBar

using MPI: MPI
using LinearAlgebra: tr

using Keldysh; kd = Keldysh

using QInchworm: SectorBlockMatrix
using QInchworm.ppgf: partition_function

using QInchworm; teval = QInchworm.topology_eval
using QInchworm.diagrammatics

using QInchworm.utility: SobolSeqWith0, next!, arbitrary_skip!
using QInchworm.utility: split_count
using QInchworm.mpi: ismaster, N_skip_and_N_samples_on_rank, all_reduce!

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

"""
$(TYPEDEF)

Inchworm algorithm input data specific to a particular expansion order.

$(TYPEDFIELDS)
"""
struct ExpansionOrderInputData
    "Expansion order"
    order::Int64
    "Number of points in the after-t_w region"
    n_pts_after::Int64
    "List of topologies contributing at this expansion order"
    topologies::Vector{teval.Topology}
    "Numbers of qMC samples (should be a power of 2)"
    N_samples::Int64
end

# http://patorjk.com/software/taag/#p=display&f=Graffiti&t=QInchWorm
const logo = raw"""________  .___              .__    __      __
\_____  \ |   | ____   ____ |  |__/  \    /  \___________  _____
 /  / \  \|   |/    \_/ ___\|  |  \   \/\/   /  _ \_  __ \/     \
/   \_/.  \   |   |  \  \___|   Y  \        (  <_> )  | \/  Y Y  \
\_____\ \_/___|___|  /\___  >___|  /\__/\  / \____/|__|  |__|_|  /
       \__>        \/     \/     \/      \/                    \/ """

#
# Inchworm / bold PPGF accumulation functions
#

raw"""
Perform one step of qMC inchworm evaluation of the bold PPGF.

The qMC integrator uses the `Root` transformation here.

Parameters
----------
expansion :  Pseudo-particle expansion problem.
c :          Integration time contour.
τ_i :        Initial time of the bold PPGF to be computed.
τ_w :        Inchworm (bare/bold splitting) time.
τ_f :        Final time of the bold PPGF to be computed.
order_data : Inchworm algorithm input data, one element per expansion order.

Returns
-------
Accumulated value of the bold pseudo-particle GF.
"""
function inchworm_step(expansion::Expansion,
                       c::kd.AbstractContour,
                       τ_i::kd.TimeGridPoint,
                       τ_w::kd.TimeGridPoint,
                       τ_f::kd.TimeGridPoint,
                       order_data::Vector{ExpansionOrderInputData};
                       tmr::TimerOutput = TimerOutput())

    t_i, t_w, t_f = τ_i.bpoint, τ_w.bpoint, τ_f.bpoint
    n_i, n_w, n_f = teval.IdentityNode(t_i), teval.InchNode(t_w), teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_w.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    orders = unique(map(od -> od.order, order_data))
    order_contribs = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)

    for od in order_data

        @timeit tmr "Bold" begin
        @timeit tmr "Order $(od.order)" begin
        @timeit tmr "Integration" begin

        order_contrib = deepcopy(zero_sector_block_matrix)

        if od.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_i, 2 => n_w, 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, fixed_nodes, tmr=tmr)
            end; @timeit tmr "Eval" begin
            order_contrib = eval(od.topologies, kd.BranchPoint[])
            end # tmr
        else
            od.N_samples <= 0 && continue
            @timeit tmr "Setup" begin

            d_after = od.n_pts_after
            d_before = 2 * od.order - od.n_pts_after

            fixed_nodes = Dict(1 => n_i, d_before + 2 => n_w, 2 * od.order + 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, od.order, fixed_nodes, tmr=tmr)

            N_skip, N_samples_on_rank = N_skip_and_N_samples_on_rank(od.N_samples)
            rank_weight = N_samples_on_rank / od.N_samples

            seq = SobolSeqWith0(2 * od.order)
            arbitrary_skip!(seq, N_skip)

            end; @timeit tmr "Eval" begin
            order_contrib = rank_weight * qmc_inchworm_integral_root(
                t -> eval(od.topologies, t),
                d_before, d_after,
                c, t_i, t_w, t_f,
                init = deepcopy(zero_sector_block_matrix),
                seq = seq,
                N = N_samples_on_rank
            )
            end; @timeit tmr "MPI all_reduce" begin
            all_reduce!(order_contrib, +)
            end # tmr
        end

        order_contribs[od.order] += order_contrib

        end; end; end # tmr

    end

    for order in orders
        set_bold_ppgf_at_order!(expansion, order, τ_i, τ_f, order_contribs[order])
    end

    return sum(values(order_contribs))
end

raw"""
Perform the first step of qMC inchworm evaluation of the bold PPGF.

This step amounts to summation of all (not just inchworm-proper) diagrams
in absence of the bold region in the integration domain.

The qMC integrator uses the `Root` transformation here.

Parameters
----------
expansion :  Pseudo-particle expansion problem.
c :          Integration time contour.
τ_i :        Initial time of the bold PPGF to be computed.
τ_f :        Final time of the bold PPGF to be computed.
order_data : Inchworm algorithm input data, one element per expansion order.

Returns
-------
Accumulated value of the bold pseudo-particle GF.
"""
function inchworm_step_bare(expansion::Expansion,
                            c::kd.AbstractContour,
                            τ_i::kd.TimeGridPoint,
                            τ_f::kd.TimeGridPoint,
                            order_data::Vector{ExpansionOrderInputData};
                            tmr::TimerOutput = TimerOutput())

    t_i, t_f = τ_i.bpoint, τ_f.bpoint
    n_i, n_f = teval.IdentityNode(t_i), teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)
    result = deepcopy(zero_sector_block_matrix)

    for od in order_data

        @timeit tmr "Bare" begin
        @timeit tmr "Order $(od.order)" begin
        @timeit tmr "Integration" begin

        order_contrib = deepcopy(zero_sector_block_matrix)

        if od.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_i, 2 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, fixed_nodes, tmr=tmr)
            end; @timeit tmr "Eval" begin
            order_contrib = eval(od.topologies, kd.BranchPoint[])
            end # tmr
        else
            od.N_samples <= 0 && continue
            @timeit tmr "Setup" begin

            d = 2 * od.order

            fixed_nodes = Dict(1 => n_i, 2 * od.order + 2 => n_f)
            eval = teval.TopologyEvaluator(expansion, od.order, fixed_nodes, tmr=tmr)

            N_skip, N_samples_on_rank = N_skip_and_N_samples_on_rank(od.N_samples)
            rank_weight = N_samples_on_rank / od.N_samples

            seq = SobolSeqWith0(d)
            arbitrary_skip!(seq, N_skip)

            end; @timeit tmr "Eval" begin
            order_contrib = rank_weight * qmc_time_ordered_integral_root(
                t -> eval(od.topologies, t),
                d,
                c, t_i, t_f,
                init = deepcopy(zero_sector_block_matrix),
                seq = seq,
                N = N_samples_on_rank
            )
            all_reduce!(order_contrib, +)
            end # tmr
        end

        set_bold_ppgf_at_order!(expansion, od.order, τ_i, τ_f, order_contrib)
        result += order_contrib

        end; end; end # tmr

    end
    
    return result
end

raw"""
Perform a complete qMC inchworm calculation of the bold PPGF on the Matsubara branch.

Result of the calculation is written into `expansion.P`.

Parameters
----------
expansion :       Pseudo-particle expansion problem.
grid :            Imaginary time grid of the bold PPGF.
orders :          List of expansion orders to be accounted for during a regular inchworm step.
orders_bare :     List of expansion orders to be accounted for during the initial inchworm step.
N_samples :       Numbers of qMC samples.
n_pts_after_max : Maximum number of points in the after-t_w region to be taken into account.
"""
function inchworm!(expansion::Expansion,
                   grid::kd.ImaginaryTimeGrid,
                   orders,
                   orders_bare,
                   N_samples::Int64;
                   n_pts_after_max::Int64 = typemax(Int64))

    tmr = TimerOutput()

    if ismaster()
        comm = MPI.COMM_WORLD
        comm_size = MPI.Comm_size(comm)
        N_split = split_count(N_samples, comm_size)

        println(logo)
        println("N_tau = ", length(grid))
        println("orders_bare = ", orders_bare)
        println("orders = ", orders)
        println("n_pts_after_max = ", n_pts_after_max)
        println("N_samples = ", N_samples)
        println("N_ranks = ", comm_size)
        println("N_samples (per rank) = ", N_split)
    end

    @assert N_samples == 0 || N_samples == 2^Int(log2(N_samples))

    # Extend expansion.P_orders to max of orders, orders_bare
    max_order = maximum([maximum(orders), maximum(orders_bare)])
    for o in 1:(max_order+1)
        push!(expansion.P_orders, kd.zero(expansion.P0))
    end

    if ismaster(); println("= Bare Diagrams ========"); end

    # First inchworm step

    order_data = ExpansionOrderInputData[]
    for order in orders_bare

        @timeit tmr "Bare" begin
        @timeit tmr "Order $(order)" begin
        @timeit tmr "Topologies" begin

        topologies = teval.get_topologies_at_order(order)

        if ismaster()
            println("Bare order $(order), N_topo $(length(topologies))")
        end

        push!(order_data, ExpansionOrderInputData(order, 2*order, topologies, N_samples))

        end; end; end # tmr "Bare" "Order" "Topologies"

    end

    if ismaster(); println("= Evaluation Bare ========"); end

    result = inchworm_step_bare(expansion,
                                grid.contour,
                                grid[1],
                                grid[2],
                                order_data, tmr=tmr)
    set_bold_ppgf!(expansion, grid[1], grid[2], result)

    if ismaster(); println("= Bold Diagrams ========"); end

    # The rest of inching

    empty!(order_data)

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
            topologies = teval.get_topologies_at_order(order, n_pts_after)

            if ismaster()
                println("Bold order $(order), n_pts_after $(n_pts_after), N_topo $(length(topologies))")
            end

            if !isempty(topologies)
                push!(order_data,
                    ExpansionOrderInputData(order, n_pts_after, topologies, N_samples)
                )
            end
        end

        end; end; end # tmr "Bold" "Order" "Topologies"

    end

    if ismaster(); println("= Evaluation Bold ========"); end

    iter = 2:length(grid)-1
    iter = ismaster() ? ProgressBar(iter) : iter

    τ_i = grid[1]
    for n in iter
        τ_w = grid[n]
        τ_f = grid[n + 1]

        result = inchworm_step(expansion, grid.contour, τ_i, τ_w, τ_f, order_data, tmr=tmr)
        set_bold_ppgf!(expansion, τ_i, τ_f, result)
    end

    if ismaster(); show(tmr); println(); end

end

#
# Calculation of two-point correlators on the Matsubara branch
#

raw"""
    Perform a calculation of a two-point correlator <A(τ) B(0)> for one value of
    the imaginary time argument τ.

    Parameters
    ----------
    expansion :    Pseudo-particle expansion problem.
    grid :         Imaginary time grid of the correlator to be computed.
    A_B_pair_idx : Index of the A/B pair in `expansion.corr_operators`.
    τ :            The imaginary time argument.
    order_data :   Accumulation input data, one element per expansion order.

    Returns
    -------
    Accummulated value of the two-point correlator.
"""
function correlator_2p(expansion::Expansion,
                       grid::kd.ImaginaryTimeGrid,
                       A_B_pair_idx::Int64,
                       τ::kd.TimeGridPoint,
                       order_data::Vector{ExpansionOrderInputData};
                       tmr::TimerOutput = TimerOutput())::ComplexF64
    t_B = grid[1].bpoint # B is always placed at τ=0
    t_A = τ.bpoint
    t_f = grid[end].bpoint

    n_B = teval.OperatorNode(t_B, A_B_pair_idx, 2)
    n_A = teval.OperatorNode(t_A, A_B_pair_idx, 1)
    n_f = teval.IdentityNode(t_f)
    @assert n_f.time.ref >= n_A.time.ref >= n_B.time.ref

    result::ComplexF64 = 0

    for od in order_data

        @timeit tmr "Order $(od.order)" begin
        @timeit tmr "Integration" begin

        order_contrib::ComplexF64 = 0

        if od.order == 0
            @timeit tmr "Setup" begin
            fixed_nodes = Dict(1 => n_B, 2 => n_A, 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, 0, fixed_nodes, tmr=tmr)
            end; @timeit tmr "Eval" begin
            order_contrib = tr(eval(od.topologies, kd.BranchPoint[]))
            end # tmr
        else
            od.N_samples <= 0 && continue
            @timeit tmr "Setup" begin

            d_after = od.n_pts_after
            d_before = 2 * od.order - od.n_pts_after

            fixed_nodes = Dict(1 => n_B, d_before + 2 => n_A, 2 * od.order + 3 => n_f)
            eval = teval.TopologyEvaluator(expansion, od.order, fixed_nodes, tmr=tmr)

            N_skip, N_samples_on_rank = N_skip_and_N_samples_on_rank(od.N_samples)
            rank_weight = N_samples_on_rank / od.N_samples

            seq = SobolSeqWith0(2 * od.order)
            arbitrary_skip!(seq, N_skip)

            end; @timeit tmr "Eval" begin
            order_contrib = rank_weight * qmc_inchworm_integral_root(
                t -> tr(eval(od.topologies, t)),
                d_before, d_after,
                grid.contour, t_B, t_A, t_f,
                init = ComplexF64(0),
                seq = seq,
                N = N_samples_on_rank
            )
            order_contrib = MPI.Allreduce(order_contrib, +, MPI.COMM_WORLD)
            end # tmr
        end

        result += order_contrib

        end; end # tmr
    end

    return result / partition_function(expansion.P)
end

raw"""
Perform a calculation of a two-point correlator <A(τ) B(0)> on the Matsubara branch.

Accumulation is performed for each pair of operators A / B passed in
`expansion.corr_operators`. Only the operators that are a single monomial in C/C^+ are
supported.

Parameters
----------
expansion :            Pseudo-particle expansion problem containing a precomputed bold PPGF
                       and pairs of operators to be used in accumulation.
grid :                 Imaginary time grid of the two-point correlator to be computed.
orders :               List of expansion orders to be accounted for.
N_samples :            Numbers of qMC samples.

Returns
-------

corr : A list of scalar-valued GF objects containing the computed correlators,
       one element per a pair in `expansion.corr_operators`.
"""
function correlator_2p(expansion::Expansion,
                       grid::kd.ImaginaryTimeGrid,
                       orders,
                       N_samples::Int64)::Vector{kd.ImaginaryTimeGF{ComplexF64, true}}

    @assert grid.contour.β == expansion.P[1].grid.contour.β

    tmr = TimerOutput()

    if ismaster()
        comm = MPI.COMM_WORLD
        comm_size = MPI.Comm_size(comm)
        N_split = split_count(N_samples, comm_size)

        println(logo)
        println("N_tau = ", length(grid))
        println("orders = ", orders)
        println("N_samples = ", N_samples)
        println("N_ranks = ", comm_size)
        println("N_samples (per rank) = ", N_split)
    end

    @assert N_samples == 0 || ispow2(N_samples)

    τ_B = grid[1]

    # Pre-compute topologies: These are common for all
    # pairs of operators in expansion.corr_operators.
    if ismaster(); println("= Topologies ========"); end
    order_data = ExpansionOrderInputData[]
    for order in orders

        @timeit tmr "Order $(order)" begin; @timeit tmr "Topologies" begin

        n_pts_after_range = (order == 0) ? (0:0) : (1:(2 * order - 1))
        for n_pts_after in n_pts_after_range
            topologies = teval.get_topologies_at_order(order, n_pts_after, with_external_arc=true)

            if !isempty(topologies)
                push!(order_data,
                      ExpansionOrderInputData(order, n_pts_after, topologies, N_samples)
                )
            end

            if ismaster()
                println("order $(order), n_pts_after $(n_pts_after), N_topo $(length(topologies))")
            end
        end

        end; end # tmr "Order" "Topologies"

    end

    # Accummulate correlators
    corr_list = kd.ImaginaryTimeGF{ComplexF64, true}[]
    for (op_pair_idx, (A, B)) in enumerate(expansion.corr_operators)

        @assert length(A) == 1 "Operator A must be a single monomial in C/C^+"
        @assert length(B) == 1 "Operator B must be a single monomial in C/C^+"

        if ismaster(); println("= Correlator <$(A),$(B)> ========"); end

        # Create a GF container to carry the result
        is_fermion(expr) = isodd(length(first(keys(expr.monomials))))
        ξ = (is_fermion(A) && is_fermion(B)) ? kd.fermionic : kd.bosonic

        push!(corr_list, kd.ImaginaryTimeGF(grid, 1, ξ, true))

        #
        # Fill in values
        #

        if ismaster(); println("= Evaluation ======== ", (A, B)); end

        # Only the 0-th order can contribute at τ_A = τ_B
        if order_data[1].order == 0
            corr_list[end][τ_B, τ_B] = correlator_2p(
                expansion,
                grid,
                op_pair_idx,
                τ_B,
                order_data[1:1],
                tmr=tmr
            )
        end

        # The rest of τ_A values
        iter = 2:length(grid)
        iter = ismaster() ? ProgressBar(iter) : iter

        for n in iter
            τ_A = grid[n]
            corr_list[end][τ_A, τ_B] = correlator_2p(
                expansion,
                grid,
                op_pair_idx,
                τ_A,
                order_data,
                tmr=tmr
            )
        end
    end

    if ismaster(); show(tmr); println(); end

    return corr_list
end

end # module inchworm
