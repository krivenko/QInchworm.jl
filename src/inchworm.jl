module inchworm

using DocStringExtensions

using ProgressBars: ProgressBar
using MPI: MPI
using LinearAlgebra: tr

using Keldysh; kd = Keldysh

using QInchworm: SectorBlockMatrix
using QInchworm.ppgf: partition_function

using QInchworm; teval = QInchworm.topology_eval
using QInchworm.diagrammatics

using QInchworm.utility: SobolSeqWith0, next!
using QInchworm.utility: inch_print
using QInchworm.utility: mpi_N_skip_and_N_samples_on_rank, split_count

using QInchworm.expansion: Expansion, set_bold_ppgf!, set_bold_ppgf_at_order!
using QInchworm.configuration: Configuration,
                               operator,
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
                       order_data::Vector{ExpansionOrderInputData})

    t_i, t_w, t_f = τ_i.bpoint, τ_w.bpoint, τ_f.bpoint
    n_i, n_w, n_f = Node(t_i), InchNode(t_w), Node(t_f)
    @assert n_f.time.ref >= n_w.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    result = deepcopy(zero_sector_block_matrix)

    for od in order_data
        order_contrib = deepcopy(zero_sector_block_matrix)
        if od.order == 0
            order_contrib = teval.eval(
                expansion, [n_f, n_w, n_i], kd.BranchPoint[], od.diagrams)
        else
            d_after = od.n_pts_after
            d_before = 2 * od.order - od.n_pts_after
            teval.update_inch_times!.(od.configurations, Ref(t_i), Ref(t_w), Ref(t_f))
            seq = SobolSeqWith0(2 * od.order)
            if od.N_samples > 0
                order_contrib = qmc_inchworm_integral_root(
                    t -> teval.eval(expansion, od.diagrams, od.configurations, t),
                    d_before, d_after,
                    c, t_i, t_w, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_samples
                )
            end
        end
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
                            order_data::Vector{ExpansionOrderInputData})

    t_i, t_f = τ_i.bpoint, τ_f.bpoint
    n_i, n_f = Node(t_i), Node(t_f)
    @assert n_f.time.ref >= n_i.time.ref

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)
    result = deepcopy(zero_sector_block_matrix)

    for od in order_data
        order_contrib = deepcopy(zero_sector_block_matrix)
        if od.order == 0
            order_contrib = teval.eval(expansion, [n_f, n_i], kd.BranchPoint[], od.diagrams)
        else
            teval.update_inch_times!.(od.configurations, Ref(t_i), Ref(t_i), Ref(t_f))
            d = 2 * od.order
            seq = SobolSeqWith0(d)
            if od.N_samples > 0
                order_contrib = qmc_time_ordered_integral_root(
                    t -> teval.eval(expansion, od.diagrams, od.configurations, t),
                    d,
                    c, t_i, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_samples
                )
            end
        end
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
expansion :            Pseudo-particle expansion problem.
orders :               List of expansion orders to be accounted for during a regular inchworm step.
orders_bare :          List of expansion orders to be accounted for during the initial inchworm step.
N_chunk :              Numbers of qMC samples taken between consecutive convergence checks.
max_chunks :           Stop accumulation after this number of unsuccessful convergence checks.
qmc_convergence_atol : Absolute tolerance level for qMC convergence checks.
"""
function inchworm_matsubara!(expansion::Expansion,
                             grid::kd.ImaginaryTimeGrid,
                             orders,
                             orders_bare,
                             N_samples::Int64)

    if inch_print()
        comm = MPI.COMM_WORLD
        comm_size = MPI.Comm_size(comm)
        N_split = split_count(N_samples, comm_size)

        println(logo)
        println("N_tau = ", length(grid))
        println("orders_bare = ", orders_bare)
        println("orders = ", orders)
        println("N_samples = ", N_samples)
        println("N_ranks = ", comm_size)
        println("N_samples (per rank) = ", N_split)
    end

    @assert N_samples == 0 || N_samples == 2^Int(log2(N_samples))

    # Extend expansion.P_orders to max of orders, orders_bare
    max_order = maximum([maximum(orders), maximum(orders_bare)])
    for o in range(1, max_order+1)
        push!(expansion.P_orders, kd.zero(expansion.P0))
    end

    if inch_print(); println("Inch step 1 (bare)"); end

    if inch_print(); println("= Bare Diagrams ========"); end
    # First inchworm step
    order_data = ExpansionOrderInputData[]
    for order in orders_bare
        topologies = teval.get_topologies_at_order(order)
        all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
        configurations, diagrams =
            teval.get_configurations_and_diagrams(
                expansion, all_diagrams, 0)

        if inch_print()
            println("order $(order), N_diag $(length(diagrams))")
            #println("diagram topologies")
            #for top in topologies
            #    println("top = $(top), ncross = $(diagrammatics.n_crossings(top)), parity = $(diagrammatics.parity(top))")
            #end
            #println("length(diagrams) = $(length(diagrams))")
            #println("length(configurations) = $(length(configurations))")
            @assert length(diagrams) == length(configurations)
        end

        if length(configurations) > 0
            push!(order_data, ExpansionOrderInputData(
                order, 2*order, diagrams, configurations, N_samples))
        end
    end

    result = inchworm_step_bare(expansion,
                                grid.contour,
                                grid[1],
                                grid[2],
                                order_data)
    set_bold_ppgf!(expansion, grid[1], grid[2], result)

    if inch_print(); println("= Bold Diagrams ========"); end
    # The rest of inching
    empty!(order_data)
    for order in orders
        #for n_pts_after = 1:max(1, 2*order-1)
        for n_pts_after = 1:1
            d_before = 2 * order - n_pts_after
            topologies = teval.get_topologies_at_order(order, n_pts_after)
            all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
            configurations, diagrams =
                teval.get_configurations_and_diagrams(
                    expansion, all_diagrams, d_before)

            if inch_print()
                println("order $(order), n_pts_after $(n_pts_after), N_diag $(length(diagrams))")
                #println("n_pts_after $(n_pts_after)")
                #println("diagram topologies")
                #for top in topologies
                #    println("top = $(top), ncross = $(diagrammatics.n_crossings(top)), parity = $(diagrammatics.parity(top))")
                #end
                #println("length(diagrams) = $(length(diagrams))")
                #println("length(configurations) = $(length(configurations))")
                @assert length(diagrams) == length(configurations)
            end

            if length(configurations) > 0
                push!(order_data, ExpansionOrderInputData(
                    order, n_pts_after, diagrams, configurations, N_samples))
            end
        end
    end

    iter = 2:length(grid)-1
    iter = inch_print() ? ProgressBar(iter) : iter

    τ_i = grid[1]
    for n in iter
        #if inch_print(); println("Inch step $n of $(length(grid)-1)"); end
        τ_w = grid[n]
        τ_f = grid[n + 1]

        result = inchworm_step(
            expansion, grid.contour, τ_i, τ_w, τ_f, order_data)
        set_bold_ppgf!(expansion, τ_i, τ_f, result)
    end
end

#
# Single-particle Matsubara GF accumulation functions
#

raw"""
    Perform accumulation of a single-particle Matsubara GF
    for one C/C^+ pair and one value of the imaginary time argument.

    Parameters
    ----------
    expansion :         Pseudo-particle expansion problem.
    grid :              Imaginary time grid of the single-particle GF to be computed.
    c_cdag_pair_idx :   Index of the C/C^+ pair in `expansion.corr_operators`.
    τ_c :               The imaginary time argument.
    order_data :        Accumulations input data, one element per expansion order.

    Returns
    -------
    Accummulated value of the single-particle Matsubara GF.
"""
function compute_gf_matsubara_point(expansion::Expansion,
                                    grid::kd.ImaginaryTimeGrid,
                                    c_cdag_pair_idx::Int64,
                                    τ_c::kd.TimeGridPoint,
                                    order_data::Vector{ExpansionOrderInputData})::ComplexF64
    t_i, t_cdag, t_c, t_f = grid[1].bpoint,
                            grid[1].bpoint, # C^+ is always placed at τ=0
                            τ_c.bpoint,
                            grid[end].bpoint

    n_i = Node(t_i)
    n_cdag = OperatorNode(t_cdag, c_cdag_pair_idx, 2)
    n_c = OperatorNode(t_c, c_cdag_pair_idx, 1)
    n_f = Node(t_f)
    @assert n_f.time.ref >= n_c.time.ref >= n_cdag.time.ref

    result::ComplexF64 = 0

    zero_sector_block_matrix = zeros(SectorBlockMatrix, expansion.ed)

    for od in order_data
        if od.order == 0
            result += tr(teval.eval(
                expansion, [n_f, n_cdag, n_c], kd.BranchPoint[], od.diagrams
            ))
        else
            d_after = od.n_pts_after
            d_before = 2 * od.order - od.n_pts_after

            teval.update_corr_times!.(od.configurations, Ref(t_cdag), Ref(t_c), Ref(t_f))

            seq = SobolSeqWith0(2 * od.order)
            if od.N_samples > 0
                result += tr(qmc_inchworm_integral_root(
                    t -> teval.eval(expansion, od.diagrams, od.configurations, t),
                    d_before, d_after,
                    grid.contour, t_i, t_c, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_samples
                   )
                )
            end
        end
    end

    result / -partition_function(expansion.P)
end

raw"""
Perform a calculation of single-particle Green's functions on the Matsubara branch.

Accumulation is performed for each annihilation/creation operator pair
passed in `expansion.corr_operators`.

Parameters
----------
expansion :            Pseudo-particle expansion problem containing a precomputed bold PPGF
                       and annihilation/creation operator pairs to be used in accumulation.
grid :                 Imaginary time grid of the single-particle GF to be computed.
orders :               List of expansion orders to be accounted for.
N_samples :            Numbers of qMC samples.

Returns
-------

gf : A list of scalar-valued single-particle GFs, one element per a pair in
     `expansion.corr_operators`.
"""
function compute_gf_matsubara(expansion::Expansion,
                              grid::kd.ImaginaryTimeGrid,
                              orders,
                              N_samples::Int64)::Vector{kd.ImaginaryTimeGF{ComplexF64, true}}
    @assert grid.contour.β == expansion.P[1].grid.contour.β

    if inch_print()
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

    τ_cdag = grid[1]

    # Pre-compute topologies and diagrams: These are common for all
    # pairs of operators in expansion.corr_operators. Some of the diagrams
    # computed here will be excluded for a specific choice of C/C^+.
    common_order_data = ExpansionOrderInputData[]
    for order in orders
        for n_pts_after = 1:max(1, 2*order-1)
            d_before = 2 * order - n_pts_after
            topologies = teval.get_topologies_at_order(order, n_pts_after)
            all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
            push!(common_order_data, ExpansionOrderInputData(
                order, n_pts_after, all_diagrams, [], N_samples))
        end
    end

    # Accummulate GFs
    gf_list = kd.ImaginaryTimeGF{ComplexF64, true}[]
    for (op_pair_idx, (c, cdag)) in enumerate(expansion.corr_operators)
        # Filter diagrams and generate lists of configurations
        order_data = ExpansionOrderInputData[]
        for od in common_order_data
            configurations, diagrams = teval.get_configurations_and_diagrams(
                expansion,
                od.diagrams,
                2 * od.order - od.n_pts_after,
                op_pair_idx = op_pair_idx
            )

            if length(configurations) > 0
                push!(order_data, ExpansionOrderInputData(
                    od.order, od.n_pts_after, diagrams, configurations, N_samples))
            end
        end

        # Create a GF container to carry the result
        push!(gf_list, kd.ImaginaryTimeGF(grid, 1, kd.fermionic, true))

        #
        # Fill in values
        #

        if inch_print()
            println("Computing matrix element ", (c, cdag))
        end

        # Only the 0-th order can contribute at τ_c = τ_cdag
        if order_data[1].order == 0
            gf_list[end][τ_cdag, τ_cdag] = compute_gf_matsubara_point(
                expansion,
                grid,
                op_pair_idx,
                τ_cdag,
                order_data[1:1]
            )
        end

        # The rest of τ_c values
        iter = 2:length(grid)
        iter = inch_print() ? ProgressBar(iter) : iter

        for n in iter
            τ_c = grid[n]
            gf_list[end][τ_c, τ_cdag] = compute_gf_matsubara_point(
                expansion,
                grid,
                op_pair_idx,
                τ_c,
                order_data
            )
        end
    end

    gf_list
end

end # module inchworm
