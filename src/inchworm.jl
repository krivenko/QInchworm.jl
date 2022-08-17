module inchworm

import Sobol: SobolSeq

import Keldysh; kd = Keldysh

import QInchworm; teval = QInchworm.topology_eval

import QInchworm.configuration: Expansion,
                                operator,
                                sector_block_matrix_from_ppgf
import QInchworm.configuration: Node, InchNode

import QInchworm.qmc_integrate: qmc_time_ordered_integral_root,
                                qmc_inchworm_integral_root

"Inchworm algorithm input data specific to a particular expansion order"
struct InchwormOrderData
    "Expansion order"
    order::Int64
    "List of diagrams contributing at this expansion order"
    diagrams::teval.Diagrams
    "Numbers of qMC samples taken between consecutive convergence checks"
    N_chunk::Int64
    "Stop accumulation after this number of unsuccessful convergence checks"
    max_chunks::Int64
    "Absolute tolerance level for qMC convergence checks"
    convergence_atol::Float64
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
            d_bare = 1
            d_bold = 2 * od.order - 1
            seq = SobolSeq(2 * od.order)
            N = 0
            order_contrib = deepcopy(zero_sector_block_matrix)
            order_contrib_prev = deepcopy(zero_sector_block_matrix)
            fill!(order_contrib_prev, Inf)

            while ((N < od.N_chunk * od.max_chunks) &&
                !isapprox(order_contrib, order_contrib_prev, atol=od.convergence_atol))
                order_contrib_prev = deepcopy(order_contrib)

                order_contrib *= N
                order_contrib += qmc_inchworm_integral_root(
                    t -> teval.eval(expansion, [n_f, n_w, n_i], t, od.diagrams),
                    d_bold, d_bare,
                    c, t_i, t_w, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_chunk
                ) * od.N_chunk
                N += od.N_chunk
                order_contrib *= 1 / N
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
            d = 2 * od.order
            seq = SobolSeq(d)
            N = 0
            order_contrib = deepcopy(zero_sector_block_matrix)
            order_contrib_prev = deepcopy(zero_sector_block_matrix)
            fill!(order_contrib_prev, Inf)

            while ((N < od.N_chunk * od.max_chunks) &&
                   !isapprox(order_contrib, order_contrib_prev, atol=od.convergence_atol))
                order_contrib_prev = deepcopy(order_contrib)

                order_contrib *= N
                order_contrib += qmc_time_ordered_integral_root(
                    t -> teval.eval(expansion, [n_f, n_i, n_i], t, od.diagrams),
                    d,
                    c, t_i, t_f,
                    init = deepcopy(zero_sector_block_matrix),
                    seq = seq,
                    N = od.N_chunk
                ) * od.N_chunk
                N += od.N_chunk
                order_contrib *= 1 / N
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
                             N_chunk::Int64,
                             max_chunks::Int64,
                             qmc_convergence_atol::Float64)
    function assign_bold_ppgf(τ_i, τ_f, result)
        for (s_i, (s_f, mat)) in result
            # Boldification must preserve the block structure
            @assert s_i == s_f
            expansion.P[s_i][τ_f, τ_i] = mat
        end
    end

    # First inchworm step
    order_data = InchwormOrderData[]
    for order in orders_bare
        topologies = teval.get_topologies_at_order(order)
        diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
        push!(order_data, InchwormOrderData(order,
                                            diagrams,
                                            N_chunk,
                                            max_chunks,
                                            qmc_convergence_atol))
    end

    result = inchworm_step_bare(expansion,
                                grid.contour,
                                grid[1].bpoint,
                                grid[2].bpoint,
                                order_data)
    assign_bold_ppgf(grid[1], grid[2], result)

    # The rest of inching
    empty!(order_data)
    for order in orders
        topologies = teval.get_topologies_at_order(order, 1)
        diagrams = teval.get_diagrams_at_order(expansion, topologies, order)
        push!(order_data, InchwormOrderData(order,
                                            diagrams,
                                            N_chunk,
                                            max_chunks,
                                            qmc_convergence_atol))
    end

    τ_i = grid[1]
    for n = 2:length(grid)-1
        τ_w = grid[n]
        τ_f = grid[n + 1]

        result = inchworm_step(expansion,
                               grid.contour,
                               τ_i.bpoint,
                               τ_w.bpoint,
                               τ_f.bpoint,
                               order_data)
        assign_bold_ppgf(τ_i, τ_f, result)
    end
end

end # module inchworm
