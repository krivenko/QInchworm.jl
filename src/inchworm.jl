module inchworm

using ProgressBars: ProgressBar
using MPI: MPI
using Printf

import Keldysh; kd = Keldysh

import QInchworm; teval = QInchworm.topology_eval
import QInchworm.diagrammatics

using  QInchworm.utility: BetterSobolSeq, next!
using  QInchworm.utility: inch_print
using QInchworm.utility: mpi_N_skip_and_N_samples_on_rank, split_count

import QInchworm.configuration: Expansion,
                                Configurations,
                                operator,
                                sector_block_matrix_from_ppgf,
                                maxabs,
                                set_bold_ppgf!

import QInchworm.configuration: Node, InchNode, SectorBlockMatrix

import QInchworm.qmc_integrate: qmc_time_ordered_integral_root,
                                qmc_inchworm_integral_root

"Inchworm algorithm input data specific to a particular expansion order"
struct InchwormOrderData
    "Expansion order"
    order::Int64
    "Number of points in the attached region"
    k_attached::Int64
    "List of diagrams contributing at this expansion order"
    diagrams::teval.Diagrams
    "Precomputed hilbert space paths"
    configurations::Configurations
    "Numbers of qMC samples (should be a power of 2)"
    N_samples::Int64
end

raw"""
Perform one step of qMC inchworm evaluation of the bold PPGF.

The qMC integrator uses the `Root` transformation here.

Parameters
----------
expansion :  Pseudo-particle expansion problem.
c :          Integration time contour.
t_i :        Initial time of the bold PPGF to be computed.
t_w :        Inchworm (bare/bold splitting) time.
t_f :        Final time of the bold PPGF to be computed.
order_data : Inchworm algorithm input data, one element per expansion order.

Returns
-------
Accummulated value of the bold pseudo-particle GF.
"""
function inchworm_step(expansion::Expansion,
                       c::kd.AbstractContour,
                       t_i::kd.BranchPoint,
                       t_w::kd.BranchPoint,
                       t_f::kd.BranchPoint,
                       order_data::Vector{InchwormOrderData})
    n_i = Node(t_i)
    n_w = InchNode(t_w)
    n_f = Node(t_f)
    @assert n_f.time.ref >= n_w.time.ref >= n_i.time.ref

    zero_sector_block_matrix = 0 * operator(expansion, n_i)

    result = deepcopy(zero_sector_block_matrix)

    for od in order_data
        if od.order == 0
            result += teval.eval(expansion, [n_f, n_w, n_i], kd.BranchPoint[], od.diagrams)
        else
            d_bare = od.k_attached
            d_bold = 2 * od.order - od.k_attached
            teval.update_inch_times!(od.configurations, t_i, t_w, t_f)
            seq = BetterSobolSeq(2 * od.order)
            order_contrib = deepcopy(zero_sector_block_matrix)
            if od.N_samples > 0
                order_contrib += qmc_inchworm_integral_root(
                    t -> teval.eval(expansion, od.diagrams, od.configurations, t),
                    d_bold, d_bare,
                    c, t_i, t_w, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_samples
                )
            end
            result += order_contrib
        end
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
t_i :        Initial time of the bold PPGF to be computed.
t_f :        Final time of the bold PPGF to be computed.
order_data : Inchworm algorithm input data, one element per expansion order.

Returns
-------
Accummulated value of the bold pseudo-particle GF.
"""
function inchworm_step_bare(expansion::Expansion,
                            c::kd.AbstractContour,
                            t_i::kd.BranchPoint,
                            t_f::kd.BranchPoint,
                            order_data::Vector{InchwormOrderData})
    n_i = Node(t_i)
    n_f = Node(t_f)
    @assert n_f.time.ref >= n_i.time.ref

    zero_sector_block_matrix = 0 * operator(expansion, n_i)
    result = deepcopy(zero_sector_block_matrix)

    for od in order_data
        if od.order == 0
            result += teval.eval(expansion, [n_f, n_i], kd.BranchPoint[], od.diagrams)
        else
            teval.update_inch_times!(od.configurations, t_i, t_i, t_f)
            d = 2 * od.order
            seq = BetterSobolSeq(d)
            order_contrib = deepcopy(zero_sector_block_matrix)
            if od.N_samples > 0
                order_contrib += qmc_time_ordered_integral_root(
                    t -> teval.eval(expansion, od.diagrams, od.configurations, t),
                    d,
                    c, t_i, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_samples
                )
            end
            result += order_contrib
        end
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

    # http://patorjk.com/software/taag/#p=display&f=Graffiti&t=QInchWorm
    logo = raw"""________  .___              .__    __      __
\_____  \ |   | ____   ____ |  |__/  \    /  \___________  _____
 /  / \  \|   |/    \_/ ___\|  |  \   \/\/   /  _ \_  __ \/     \
/   \_/.  \   |   |  \  \___|   Y  \        (  <_> )  | \/  Y Y  \
\_____\ \_/___|___|  /\___  >___|  /\__/\  / \____/|__|  |__|_|  /
       \__>        \/     \/     \/      \/                    \/ """

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

    if inch_print(); println("Inch step 1 (bare)"); end

    if inch_print(); println("= Bare Diagrams ========"); end
    # First inchworm step
    order_data = InchwormOrderData[]
    for order in orders_bare
        topologies = teval.get_topologies_at_order(order)
        all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
        configurations, diagrams =
            teval.get_configurations_and_diagrams(
                expansion, all_diagrams, 0)
        
        if inch_print()
            println("order $(order)")
            println("diagram topologies")
            for top in topologies
                println("top = $(top), ncross = $(diagrammatics.n_crossings(top)), parity = $(diagrammatics.parity(top))")
            end
            println("length(diagrams) = $(length(diagrams))")
            println("length(configurations) = $(length(configurations))")
            @assert length(diagrams) == length(configurations)
        end

        if length(configurations) > 0
            push!(order_data, InchwormOrderData(
                order, 2*order, diagrams, configurations, N_samples))
        end
    end

    result = inchworm_step_bare(expansion,
                                grid.contour,
                                grid[1].bpoint,
                                grid[2].bpoint,
                                order_data)
    set_bold_ppgf!(expansion, grid[1], grid[2], result)

    if inch_print(); println("= Bold Diagrams ========"); end
    # The rest of inching
    empty!(order_data)
    for order in orders
        #for k_attached = 1:max(1, 2*order-1)
        for k_attached = 1:1
            d_bold = 2 * order - k_attached
            topologies = teval.get_topologies_at_order(order, k_attached)
            all_diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
            configurations, diagrams =
                teval.get_configurations_and_diagrams(
                    expansion, all_diagrams, d_bold)

            if inch_print()
                println("order $(order)")
                println("k_attached $(k_attached)")
                println("diagram topologies")
                for top in topologies
                    println("top = $(top), ncross = $(diagrammatics.n_crossings(top)), parity = $(diagrammatics.parity(top))")
                end
                println("length(diagrams) = $(length(diagrams))")
                println("length(configurations) = $(length(configurations))")
                @assert length(diagrams) == length(configurations)
            end

            if length(configurations) > 0
                push!(order_data, InchwormOrderData(
                    order, k_attached, diagrams, configurations, N_samples))
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

        result = inchworm_step(expansion,
                               grid.contour,
                               τ_i.bpoint,
                               τ_w.bpoint,
                               τ_f.bpoint,
                               order_data)
        set_bold_ppgf!(expansion, τ_i, τ_f, result)
    end
end

end # module inchworm
