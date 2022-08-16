module inchworm

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED

import QInchworm; teval = QInchworm.topology_eval

import QInchworm.configuration: Expansion,
                                operator_to_sector_block_matrix,
                                sector_block_matrix_from_ppgf
import QInchworm.configuration: Node, InchNode

import QInchworm.qmc_integrate: qmc_inchworm_integral_root

raw"""
Perform one step of qMC inchworm evaluation of the bold PPGF.

The qMC integrator uses the `Root` transformation here.

Parameters
----------
expansion : Pseudo-particle expansion problem.
c : Integration time contour.
t_i : Initial time of the bold PPGF to be computed.
t_w : Inchworm (bare/bold splitting) time.
t_f : Final time of the bold PPGF to be computed.
orders : Inchworm perturbation orders to sum over.
N_samples : Numbers of qMC samples, one value per perturbation order.

Returns
-------
Accummulated value of the bold pseudo-particle GF.
"""
function inchworm_step(expansion::Expansion,
                       c::kd.AbstractContour,
                       t_i::kd.BranchPoint,
                       t_w::kd.BranchPoint,
                       t_f::kd.BranchPoint,
                       orders,
                       N_samples::Vector{Int})
    n_i = Node(t_i)
    n_w = InchNode(t_w)
    n_f = Node(t_f)
    @assert n_f.time.ref >= n_w.time.ref >= n_i.time.ref

    zero_sector_block_matrix = operator_to_sector_block_matrix(
        expansion,
        ked.Operators.RealOperatorExpr()
    )

    result = deepcopy(zero_sector_block_matrix)
    for (order, N) in zip(orders, N_samples)
        if t_i == t_w # First inching step without the bold part
            d_bare = 2*order
            d_bold = 0
        else # Both bold and bare parts are present
            # QUESTION: For now, I fix d_bare = 1
            d_bare = 1
            d_bold = 2*order - 1
        end
        topologies = teval.get_topologies_at_order(order, d_bare)
        diagrams = teval.get_diagrams_at_order(expansion, topologies, order)

        if order == 0
            result += teval.eval(expansion, [n_f, n_w, n_i], kd.BranchPoint[], diagrams)
        else
            result += qmc_inchworm_integral_root(
                t -> teval.eval(expansion, [n_f, n_w, n_i], t, diagrams),
                d_bold, d_bare,
                c, t_i, t_w, t_f,
                init = deepcopy(zero_sector_block_matrix),
                N = N
            )
        end
    end
    result
end

end # module inchworm
