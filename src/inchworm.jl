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

using QInchworm.utility: SobolSeqWith0, next!
using QInchworm.utility: split_count
using QInchworm.mpi: ismaster

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
    "List of diagrams contributing at this expansion order"
    diagrams::Vector{teval.Diagram}
    "Precomputed hilbert space paths"
    configurations::Vector{Configuration}
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
Accummulated value of the bold pseudo-particle GF.
"""
function inchworm_step(expansion::Expansion,
                       c::kd.AbstractContour,
                       τ_i::kd.TimeGridPoint,
                       τ_w::kd.TimeGridPoint,
                       τ_f::kd.TimeGridPoint,
                       order_data::Vector{ExpansionOrderInputData};
                       tmr::TimerOutput = TimerOutput())

    t_i, t_w, t_f = τ_i.bpoint, τ_w.bpoint, τ_f.bpoint
    n_i, n_w, n_f = Node(t_i), InchNode(t_w), Node(t_f)
    @assert n_f.time.ref >= n_w.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    result = deepcopy(zero_sector_block_matrix)

    orders = unique(map(od -> od.order, order_data))
    #order_contribs = Dict(o => deepcopy(zero_sector_block_matrix) for o in orders)

    for od in order_data

        order_contrib = deepcopy(zero_sector_block_matrix)
        
        for diagram in od.diagrams

            diagrams = [diagram]
            
        @timeit tmr "Bold" begin
        @timeit tmr "Order $(od.order)" begin;
        @timeit tmr "Configurations" begin;

        #empty!(od.configurations)
        configurations, diagrams = teval.get_configurations_and_diagrams_serial(
            expansion, diagrams, 2 * od.order - od.n_pts_after)
            #expansion, od.diagrams, 2 * od.order - od.n_pts_after)
        #append!(od.configurations, configurations)

        end; end; end # tmr

        @timeit tmr "Bold" begin
        @timeit tmr "Order $(od.order)" begin
        @timeit tmr "Integration" begin;

        #set_initial_node_time!.(od.configurations, Ref(t_i))
        #set_inchworm_node_time!.(od.configurations, Ref(t_w))
        #set_final_node_time!.(od.configurations, Ref(t_f))

        set_initial_node_time!.(configurations, Ref(t_i))
        set_inchworm_node_time!.(configurations, Ref(t_w))
        set_final_node_time!.(configurations, Ref(t_f))
            
        if od.order == 0
            #order_contribs[od.order]
            order_contrib += teval.eval(
                expansion, diagrams, configurations, kd.BranchPoint[])
                #expansion, diagrams, od.configurations, kd.BranchPoint[])
                #expansion, od.diagrams, od.configurations, kd.BranchPoint[])
        else
            d_after = od.n_pts_after
            d_before = 2 * od.order - od.n_pts_after

            seq = SobolSeqWith0(2 * od.order)
            if od.N_samples > 0
                order_contrib += qmc_inchworm_integral_root(
                    #t -> teval.eval(expansion, od.diagrams, od.configurations, t),
                    #t -> teval.eval(expansion, diagrams, od.configurations, t),
                    t -> teval.eval(expansion, diagrams, configurations, t),
                    d_before, d_after,
                    c, t_i, t_w, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_samples
                )
            end
        end

        end; end; end # tmr

        end # for od.diagrams

        #if true
        if od.order > 0
            if isa(order_contrib, Dict)
                for (s_i, (s_f, mat)) in order_contrib
                    mat[:] = MPI.Allreduce(mat, +, MPI.COMM_WORLD)
                end
            else
                # -- For supporting the qmc_integrate.jl tests
                order_contrib = MPI.Allreduce(order_contrib, +, MPI.COMM_WORLD)
            end
        end
        #end

        #if ismaster()
        #    @show od.order
        #    for (s_i, (s_f, mat)) in order_contrib
        #        @show s_i, s_f, mat
        #    end
        #end

        #if od.order > 0
        #    exit()
        #end
        
        #order_contribs[od.order] = order_contrib
        #set_bold_ppgf_at_order!(expansion, order, τ_i, τ_f, order_contribs[order])
        set_bold_ppgf_at_order!(expansion, od.order, τ_i, τ_f, order_contrib)
        result += order_contrib

    end

    result
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
Accummulated value of the bold pseudo-particle GF.
"""
function inchworm_step_bare(expansion::Expansion,
                            c::kd.AbstractContour,
                            τ_i::kd.TimeGridPoint,
                            τ_f::kd.TimeGridPoint,
                            order_data::Vector{ExpansionOrderInputData};
                            tmr::TimerOutput = TimerOutput())

    t_i, t_f = τ_i.bpoint, τ_f.bpoint
    n_i, n_f = Node(t_i), Node(t_f)
    @assert n_f.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)
    result = deepcopy(zero_sector_block_matrix)

    for od in order_data

        order_contrib = deepcopy(zero_sector_block_matrix)

        for diagram in od.diagrams

            diagrams = [diagram]
            
        @timeit tmr "Bare" begin
        @timeit tmr "Order $(od.order)" begin;
        @timeit tmr "Configurations" begin;

        #empty!(od.configurations)
        configurations, diagrams =
            #teval.get_configurations_and_diagrams(expansion, od.diagrams, nothing)
            teval.get_configurations_and_diagrams_serial(expansion, diagrams, nothing)
        #append!(od.configurations, configurations)

        end; end; end # tmr

        @timeit tmr "Bare" begin
        @timeit tmr "Order $(od.order)" begin
        @timeit tmr "Integration" begin;

        #set_initial_node_time!.(od.configurations, Ref(t_i))
        #set_final_node_time!.(od.configurations, Ref(t_f))

        set_initial_node_time!.(configurations, Ref(t_i))
        set_final_node_time!.(configurations, Ref(t_f))

        if od.order == 0
            order_contrib += teval.eval(
                #expansion, od.diagrams, od.configurations, kd.BranchPoint[])
                expansion, diagrams, configurations, kd.BranchPoint[])
        else
            d = 2 * od.order
            seq = SobolSeqWith0(d)
            if od.N_samples > 0
                order_contrib += qmc_time_ordered_integral_root(
                    #t -> teval.eval(expansion, od.diagrams, od.configurations, t),
                    t -> teval.eval(expansion, diagrams, configurations, t),
                    d,
                    c, t_i, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_samples
                )
            end
        end

        end; end; end # tmr

        end # for od.diagrams

        #if true
        if od.order > 0
            if isa(order_contrib, Dict)
                for (s_i, (s_f, mat)) in order_contrib
                    mat[:] = MPI.Allreduce(mat, +, MPI.COMM_WORLD)
                end
            else
                # -- For supporting the qmc_integrate.jl tests
                order_contrib = MPI.Allreduce(order_contrib, +, MPI.COMM_WORLD)
            end
        end
        #end # false

        #if ismaster()
        #    @show od.order
        #    for (s_i, (s_f, mat)) in order_contrib
        #        @show s_i, s_f, mat
        #    end
        #end
        
        set_bold_ppgf_at_order!(expansion, od.order, τ_i, τ_f, order_contrib)
        result += order_contrib

    end
    result
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

        @timeit tmr "Bare" begin; @timeit tmr "Order $(order)" begin; @timeit tmr "Topologies" begin

        #local_tmr = TimerOutput()
        #@timeit local_tmr "Order $(order)" begin;
        #@timeit local_tmr "Topologies" begin

        #@time topologies = teval.get_topologies_at_order(order)
        topologies = teval.get_topologies_at_order(order)

        if ismaster(); println("Order $(order) N_topo $(length(topologies))"); end

        #end; end; if ismaster(); show(local_tmr); println(); end

        #@timeit local_tmr "Order $(order)" begin;
        #@timeit local_tmr "Diagrams" begin

        end; end; end # tmr "Bare" "Order"

        @timeit tmr "Bare" begin; @timeit tmr "Order $(order)" begin; @timeit tmr "Diagrams" begin

        #all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
            
        #@time diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
        diagrams = teval.get_diagrams_at_order(expansion, topologies, order)

        #if ismaster(); println("Order $(order) N_diag $(length(all_diagrams))"); end

        #end; end; if ismaster(); show(local_tmr); println(); end

        #@timeit local_tmr "Order $(order)" begin;
        #@timeit local_tmr "Diagrams non-zero" begin

        end; end; end # tmr "Bare" "Order"

        @timeit tmr "Bare" begin; @timeit tmr "Order $(order)" begin; @timeit tmr "Diag/Conf" begin

        #configurations_dummy, diagrams =
        #    teval.get_configurations_and_diagrams(
        #        expansion, all_diagrams, nothing, return_configurations=false)

        #@time configurations_dummy, diagrams =
        configurations_dummy, diagrams =
            teval.get_configurations_and_diagrams_from_topologies(
                expansion, topologies, order, nothing, return_configurations=false)

        #end; end; if ismaster(); show(local_tmr); println(); end

        if ismaster()
            #println("Bare Order $(order), N_diag $(length(all_diagrams)) -> $(length(diagrams))")
            println("Bare Order $(order), N_topo $(length(topologies)), N_diag $(length(diagrams))")
        end

        if length(diagrams) > 0
            push!(order_data, ExpansionOrderInputData(
                order, 2*order, diagrams, [], N_samples))
                #order, 2*order, diagrams, configurations, N_samples))
        end

        end; end; end # tmr "Bare" "Order"

        #if ismaster(); show(tmr); println(); end    

    end

    #exit()
    
    if ismaster(); println("= Evaluation Bare ========"); end

    result = inchworm_step_bare(expansion,
                                grid.contour,
                                grid[1],
                                grid[2],
                                order_data, tmr=tmr)
    set_bold_ppgf!(expansion, grid[1], grid[2], result)

    if ismaster(); show(tmr); println(); end

    if ismaster(); println("= Bold Diagrams ========"); end

    # The rest of inching

    empty!(order_data)

    for order in orders

        @timeit tmr "Bold" begin; @timeit tmr "Order $(order)" begin; @timeit tmr "Diagrams" begin

        if order == 0
            n_pts_after_range = 0:0
        else
            n_pts_after_range = 1:min(2 * order - 1, n_pts_after_max)
        end

        for n_pts_after in n_pts_after_range
            d_before = 2 * order - n_pts_after
            topologies = teval.get_topologies_at_order(order, n_pts_after)

            #all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
            #configurations_dummay, diagrams = teval.get_configurations_and_diagrams(
            #    expansion, all_diagrams, d_before, return_configurations=false)

            configurations_dummay, diagrams = teval.get_configurations_and_diagrams_from_topologies(
                expansion, topologies, order, d_before, return_configurations=false)

            if ismaster()
                #println("Bold order $(order), n_pts_after $(n_pts_after), N_diag $(length(all_diagrams)) -> $(length(diagrams))")
                println("Bold order $(order), n_pts_after $(n_pts_after), N_topo $(length(topologies)), N_diag $(length(diagrams))")
            end

            if length(diagrams) > 0
                push!(order_data, ExpansionOrderInputData(
                    order, n_pts_after, diagrams, [], N_samples))
                #order, n_pts_after, diagrams, configurations, N_samples))
            end
        end

        end; end; end # tmr "Bold" "Order" "Diagrams"

    end

    if ismaster(); println("= Evaluation Bold ========"); end

    iter = 2:length(grid)-1
    iter = ismaster() ? ProgressBar(iter) : iter

    τ_i = grid[1]
    for n in iter
        #if ismaster(); println("Inch step $n of $(length(grid)-1)"); end
        τ_w = grid[n]
        τ_f = grid[n + 1]

        result = inchworm_step(
            expansion, grid.contour, τ_i, τ_w, τ_f, order_data, tmr=tmr)
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
    Accummulated value of the single-particle Matsubara GF.
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

    n_B = OperatorNode(t_B, A_B_pair_idx, 2)
    n_A = OperatorNode(t_A, A_B_pair_idx, 1)
    n_f = Node(t_f)
    @assert n_f.time.ref >= n_A.time.ref >= n_B.time.ref

    result::ComplexF64 = 0

    for od in order_data

        @timeit tmr "Order $(od.order)" begin; @timeit tmr "Integration" begin

        set_operator_node_time!.(od.configurations, 1, Ref(t_B))
        set_operator_node_time!.(od.configurations, 2, Ref(t_A))
        set_final_node_time!.(od.configurations, Ref(t_f))

        if od.order == 0
            result += tr(teval.eval(
                expansion, od.diagrams, od.configurations, kd.BranchPoint[]))
        else
            d_after = od.n_pts_after
            d_before = 2 * od.order - od.n_pts_after

            seq = SobolSeqWith0(2 * od.order)
            if od.N_samples > 0
                result += qmc_inchworm_integral_root(
                    t -> tr(teval.eval(expansion, od.diagrams, od.configurations, t)),
                    d_before, d_after,
                    grid.contour, t_B, t_A, t_f,
                    init = ComplexF64(0),
                    seq = seq,
                    N = od.N_samples
                )
            end
        end

        end; end # tmr
    end

    if true
    if od.order > 0
    if isa(result, Dict)
        for (s_i, (s_f, mat)) in result
            mat[:] = MPI.Allreduce(mat, +, MPI.COMM_WORLD)
        end
    else
        # -- For supporting the qmc_integrate.jl tests
        result = MPI.Allreduce(result, +, MPI.COMM_WORLD)
    end
    end    
    end # false

    result / partition_function(expansion.P)
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
function correlator_2p_depr(expansion::Expansion,
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

    @assert N_samples == 0 || N_samples == 2^Int(log2(N_samples))

    τ_B = grid[1]

    # Pre-compute topologies and diagrams: These are common for all
    # pairs of operators in expansion.corr_operators. Some of the diagrams
    # computed here will be excluded for a specific choice of the operators A and B.
    if ismaster(); println("= Diagrams ========"); end
    common_order_data = ExpansionOrderInputData[]
    for order in orders

        @timeit tmr "Order $(order)" begin; @timeit tmr "Diagrams" begin

        n_pts_after_range = (order == 0) ? (0:0) : (1:(2 * order - 1))
        for n_pts_after in n_pts_after_range
            d_before = 2 * order - n_pts_after
            topologies = teval.get_topologies_at_order(order, n_pts_after, with_1k_arc=true)
            all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)

            push!(common_order_data, ExpansionOrderInputData(
                order, n_pts_after, all_diagrams, [], N_samples))

            if ismaster()
                println("order $(order), n_pts_after $(n_pts_after), N_diag $(length(all_diagrams))")
            end
        end

        end; end # tmr "Order" "Diagrams"

    end

    # Accummulate correlators
    corr_list = kd.ImaginaryTimeGF{ComplexF64, true}[]
    for (op_pair_idx, (A, B)) in enumerate(expansion.corr_operators)

        @assert length(A) == 1 "Operator A must be a single monomial in C/C^+"
        @assert length(B) == 1 "Operator B must be a single monomial in C/C^+"

        if ismaster(); println("= Correlator <$(A),$(B)> ========"); end

        # Filter diagrams and generate lists of configurations
        order_data = ExpansionOrderInputData[]
        for od in common_order_data

            @timeit tmr "Order $(od.order)" begin; @timeit tmr "Configurations" begin

            configurations, diagrams = teval.get_configurations_and_diagrams(
                expansion,
                od.diagrams,
                2 * od.order - od.n_pts_after,
                op_pair_idx = op_pair_idx
            )

            if ismaster()
                println("order $(od.order) n_pts_after $(od.n_pts_after) N_diag $(length(od.diagrams)) -> $(length(diagrams))")
            end

            if length(configurations) > 0
                push!(order_data, ExpansionOrderInputData(
                    od.order, od.n_pts_after, diagrams, configurations, N_samples))
            end

            end; end # tmr "Order" "Configurations"
        end

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

    @assert N_samples == 0 || N_samples == 2^Int(log2(N_samples))

    τ_B = grid[1]

    # Accummulate GFs

    corr_list = kd.ImaginaryTimeGF{ComplexF64, true}[]

    for (op_pair_idx, (A, B)) in enumerate(expansion.corr_operators)

        @assert length(A) == 1 "Operator A must be a single monomial in C/C^+"
        @assert length(B) == 1 "Operator B must be a single monomial in C/C^+"

        if ismaster(); println("= Correlator <$(A),$(B)> ========"); end

        # Create a GF container to carry the result
        is_fermion(expr) = isodd(length(first(keys(expr.monomials))))
        ξ = (is_fermion(A) && is_fermion(B)) ? kd.fermionic : kd.bosonic

        push!(corr_list, kd.ImaginaryTimeGF(grid, 1, ξ, true))

        for order in orders

            n_pts_after_range = (order == 0) ? (0:0) : (1:(2 * order - 1))

            for n_pts_after in n_pts_after_range

                @timeit tmr "Order $(order)" begin; @timeit tmr "Diag + Conf" begin

                d_before = 2 * order - n_pts_after
                topologies = teval.get_topologies_at_order(order, n_pts_after, with_1k_arc=true)

                #all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
                #configurations, diagrams = teval.get_configurations_and_diagrams(
                #    expansion, all_diagrams, 2 * order - n_pts_after, op_pair_idx=op_pair_idx)

                configurations, diagrams = teval.get_configurations_and_diagrams_from_topologies(
                    expansion, topologies, order, 2 * order - n_pts_after, op_pair_idx=op_pair_idx)

                if ismaster()
                    #println("order $(order), n_pts_after $(n_pts_after), N_diag $(length(all_diagrams)) -> $(length(diagrams))")
                    println("order $(order), n_pts_after $(n_pts_after), N_topo $(length(topologies)), N_diag $(length(diagrams))")
                end

                order_data = ExpansionOrderInputData[]

                if length(configurations) > 0
                    push!(order_data, ExpansionOrderInputData(
                        order, n_pts_after, diagrams, configurations, N_samples))
                end

                end; end # tmr

                #
                # Fill in values
                #

                # Only the 0-th order can contribute at τ_A = τ_B
                if length(order_data) > 0 && order_data[1].order == 0
                    corr_list[end][τ_B, τ_B] += correlator_2p(
                        expansion, grid, op_pair_idx, τ_B, order_data[1:1], tmr=tmr)
                end

                # The rest of τ_A values
                iter = 2:length(grid)
                iter = ismaster() ? ProgressBar(iter) : iter

                for n in iter
                    τ_A = grid[n]
                    corr_list[end][τ_A, τ_B] += correlator_2p(
                        expansion, grid, op_pair_idx, τ_A, order_data, tmr=tmr)
                end
            end

            #end; end # tmr "Order" "Diagrams"

        end

    end

    if ismaster(); show(tmr); println(); end

    return corr_list
end

end # module inchworm
