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
# Authors: Igor Krivenko, Hugo U. R. Strand, Joseph Kleinhenz

"""
Evaluator for strong-coupling expansion diagrams of a specific topology.
"""
module topology_eval

using DocStringExtensions

using TimerOutputs: TimerOutput, @timeit

using Keldysh; kd = Keldysh

using QInchworm.sector_block_matrix: SectorBlockMatrix
using QInchworm.expansion: Expansion
using QInchworm.diagrammatics: Topology,
                               is_doubly_k_connected,
                               generate_topologies

using QInchworm.utility: LazyMatrixProduct, eval!

using QInchworm.barycentric_interp: barycentric_interpolate!

Base.isless(t1::kd.BranchPoint, t2::kd.BranchPoint) = !kd.heaviside(t1, t2)

"""
$(TYPEDEF)

Node kind classification using `@enum`.

Possible values: `pair_flag`, `identity_flag`, `inch_flag`, `operator_flag`.
"""
@enum NodeKind pair_flag=1 identity_flag=2 inch_flag=3 operator_flag=4

"""
    $(TYPEDEF)

Node in the atomic propagator backbone of a strong-coupling diagram.

# Fields
$(TYPEDFIELDS)
"""
struct Node
    "Kind of the node"
    kind::NodeKind
    "Index for pair interaction arc"
    arc_index::Int64
    "Index of operator"
    operator_index::Int64
end

"""
    $(TYPEDEF)

Node in the atomic propagator backbone of a strong-coupling diagram fixed at a certain
contour time point.

# Fields
$(TYPEDFIELDS)
"""
struct FixedNode
    "Reference to operator"
    node::Node
    "Contour time point"
    time::kd.BranchPoint
end

"""
    $(TYPEDSIGNATURES)

Return a node that serves as an end of a pair interaction arc fixed at time `time`.
"""
function PairNode(time::kd.BranchPoint)::FixedNode
    return FixedNode(Node(pair_flag, -1, -1), time)
end

"""
    $(TYPEDSIGNATURES)

Return a fixed node at time `time` with an associated identity operator.
"""
function IdentityNode(time::kd.BranchPoint)::FixedNode
    return FixedNode(Node(identity_flag, -1, -1), time)
end

"""
    $(TYPEDSIGNATURES)

Return a fixed 'inch' node at time `time` with an associated identity operator.
"""
function InchNode(time::kd.BranchPoint)::FixedNode
    return FixedNode(Node(inch_flag, -1, -1), time)
end

"""
    $(TYPEDSIGNATURES)

Return a fixed operator node at time `time` with an associated operator.
The actual operator is stored in an [`Expansion`](@ref) structure and is uniquely identified
by the pair `(operator_pair_index, operator_index)`.
"""
function OperatorNode(time::kd.BranchPoint,
                      operator_pair_index::Int64,
                      operator_index::Int64)::FixedNode
    return FixedNode(Node(operator_flag, operator_pair_index, operator_index), time)
end

"""
    $(TYPEDEF)

The evaluation engine for the strong-coupling expansion diagrams.

In the following, a sequence of [`Node`'s](@ref Node) contributing to a diagram of a certain
topology is referred to as 'configuration'.

# Fields
$(TYPEDFIELDS)
"""
struct TopologyEvaluator

    "Pseudo-particle expansion problem"
    exp::Expansion

    "Configuration as a list of nodes arranged in the contour order"
    conf::Vector{Node}

    "Contour positions of nodes in the configuration"
    times::Vector{kd.BranchPoint}

    "Correspondence of node positions within a topology and a configuration"
    top_to_conf_pos::Vector{Int64}

    "Positions of variable time (non-fixed) nodes within a configuration"
    var_time_pos::Vector{Int64}

    "Must the bold PPGFs be used?"
    use_bold_prop::Bool

    """
    PPGFs evaluated at all relevant pairs of time arguments.
    `ppgf_mats[i, s]` is the `s`-th diagonal block of `exp.P` (or `exp.P0`) evaluated at
    the pair of time points ``(t_{i+1}, t_i)``.
    """
    ppgf_mats::Array{Matrix{ComplexF64}, 2}

    """
    Pair interaction arcs evaluated at all relevant pairs of time arguments.

    `pair_ints[a, p]` is the propagator from `exp.pairs[p]` evaluated at the pair of time
    points corresponding to the `a`-th arc in a topology.
    """
    pair_ints::Array{ComplexF64, 2}

    """
    Indices of pair interactions within `exp.pairs` assigned to each
    interaction arc in a topology.
    """
    selected_pair_ints::Vector{Int64}

    """Pre-allocated container for per-topology evaluation results"""
    top_result_mats::Vector{Matrix{ComplexF64}}

    """Pre-allocated matrix product evaluator"""
    matrix_prod::LazyMatrixProduct{ComplexF64}

    """Internal performance timer"""
    tmr::TimerOutput

    @doc """
        $(TYPEDSIGNATURES)

    # Parameters
    - `exp`:           Strong-coupling expansion problem.
    - `order`:         Expansion order of the diagrams (the number of interaction arcs).
    - `use_bold_prop`: Must the bold PPGFs be used in the diagrams?
    - `fixed_nodes`:   List of fixed nodes in a configuration along with their positions.
    - `tmr`:           Internal performance timer.
    """
    function TopologyEvaluator(exp::Expansion,
                               order::Int,
                               use_bold_prop::Bool,
                               fixed_nodes::Dict{Int, FixedNode};
                               tmr::TimerOutput = TimerOutput())
        n_nodes = 2 * order + count(n -> n.second.node.kind != pair_flag, fixed_nodes)
        @assert maximum(keys(fixed_nodes)) <= n_nodes

        # Prepare a skeleton of the configuration by placing only the fixed nodes
        conf = Vector{Node}(undef, n_nodes)
        times = Vector{kd.BranchPoint}(undef, n_nodes)

        for (pos, fn) in fixed_nodes
            conf[pos] = fn.node
            times[pos] = fn.time
        end

        # Build the `top_to_conf_pos` map.
        # We need the reverse() here because the orders of nodes in a topology and in a
        # configurations are reversed. Fixed pair nodes still belong to a topology.
        top_to_conf_pos = [
            pos for pos in reverse(1:n_nodes) if
            !(haskey(fixed_nodes, pos) && fixed_nodes[pos].node.kind != pair_flag)
        ]

        # Build the `var_time_pos` map.
        var_time_pos = [pos for pos in reverse(1:n_nodes) if !haskey(fixed_nodes, pos)]

        ppgf_mats = [Matrix{ComplexF64}(undef, norbitals(p), norbitals(p))
                     for _ in 1:(n_nodes-1), p in exp.P]
        pair_ints = Array{ComplexF64}(undef, order, length(exp.pairs))
        selected_pair_ints = Vector{Int64}(undef, order)
        top_result_mats = [zeros(ComplexF64, norbitals(p), norbitals(p)) for p in exp.P]
        matrix_prod = LazyMatrixProduct(ComplexF64, 2 * n_nodes - 1)

        return new(exp,
                   conf,
                   times,
                   top_to_conf_pos,
                   var_time_pos,
                   use_bold_prop,
                   ppgf_mats,
                   pair_ints,
                   selected_pair_ints,
                   top_result_mats,
                   matrix_prod,
                   tmr)
    end
end

"""
    $(TYPEDSIGNATURES)

Given a single diagram topology, evaluate contribution from all relevant configurations
assuming that non-fixed nodes are located at contour time points `times`.
"""
function (eval::TopologyEvaluator)(topology::Topology,
    times::Vector{kd.BranchPoint})::SectorBlockMatrix
    return eval([topology], times)
end

"""
    $(TYPEDSIGNATURES)

Given a list of diagram topologies, evaluate contribution from all relevant configurations
assuming that non-fixed nodes are located at contour time points `times`.
"""
function (eval::TopologyEvaluator)(topologies::Vector{Topology},
                                   times::Vector{kd.BranchPoint})::SectorBlockMatrix

    # Update non-fixed elements of eval.times
    eval.times[eval.var_time_pos] = times

    # Pre-compute eval.ppgf_mats
    for i in axes(eval.ppgf_mats, 1)
        time_i = eval.times[i]
        time_f = eval.times[i + 1]

        # Tackle time ordering violations caused by rounding errors
        if time_f < time_i
            time_f = time_i
        end

        for s in axes(eval.ppgf_mats, 2)
            if eval.use_bold_prop
                #kd.interpolate!(eval.ppgf_mats[i, s], eval.exp.P[s], time_f, time_i)
                barycentric_interpolate!(eval.ppgf_mats[i, s], eval.exp.interpolation_order,
                                         eval.exp.P[s], time_f, time_i)
            else
                kd.interpolate!(eval.ppgf_mats[i, s], eval.exp.P0[s], time_f, time_i)
            end
            eval.ppgf_mats[i, s] *= im
        end
    end

    result_mats = [zeros(ComplexF64, norbitals(p), norbitals(p)) for p in eval.exp.P]

    for top in topologies # TODO: Parallelization opportunity I

        @assert length(top.pairs) == length(eval.selected_pair_ints)

        # Pre-compute eval.pair_ints and place pair interaction nodes into the configuration
        for (a, arc) in enumerate(top.pairs)
            pos_head = eval.top_to_conf_pos[arc[2]]
            pos_tail = eval.top_to_conf_pos[arc[1]]
            @assert pos_tail > pos_head

            eval.conf[pos_head] = Node(pair_flag, a, 1)
            eval.conf[pos_tail] = Node(pair_flag, a, 2)

            time_i = eval.times[pos_head]
            time_f = eval.times[pos_tail]
            # Tackle time ordering violations caused by rounding errors
            if time_f < time_i
                time_f = time_i
            end

            for (p, int_pair) in enumerate(eval.exp.pairs)
                eval.pair_ints[a, p] = im * int_pair.propagator(time_f, time_i)
            end
        end

        @timeit eval.tmr "Tree traversal" begin

        fill!.(eval.top_result_mats, 0.0)

        # Traverse the configuration tree for each initial subspace
        for s_i in eachindex(eval.exp.P) # TODO: Parallelization opportunity II
            @assert eval.matrix_prod.n_mats == 0
            _traverse_configuration_tree!(eval,
                                          1,
                                          s_i, s_i,
                                          ComplexF64(1))
        end

        result_mats .+= -im * top.parity * (-1)^top.order * eval.top_result_mats

        end # tmr
    end

    return Dict(s => (s, mat) for (s, mat) in enumerate(result_mats))
end

"""
    $(TYPEDSIGNATURES)

Recursively traverse a tree of all configurations stemming from a given topology and
contributing to the quantity of interest. The result is accumulated in
`eval.top_result_mats`.

# Parameters
- `eval`:            Evaluator object.
- `pos`:             Position of the currently processed [`Node`](@ref) in the
                     configuration.
- `s_i`:             Left block index of the matrix representation of the current node.
- `s_f`:             Right block index expected at the final node.
- `pair_int_weight`: Current weight of the pair interaction contribution.
"""
function _traverse_configuration_tree!(eval::TopologyEvaluator,
                                       pos::Int,
                                       s_i::Int,
                                       s_f::Int,
                                       pair_int_weight::ComplexF64)

    # Are we at a leaf?
    if pos > length(eval.conf)
        if s_i == s_f # Is the resulting configuration block-diagonal?
            eval.top_result_mats[s_i] .+= pair_int_weight * eval!(eval.matrix_prod)
        end
        return
    end

    node = eval.conf[pos] # Current node

    if node.kind == pair_flag

        if node.operator_index == 1 # Head of an interaction arc

            # Loop over all interaction pairs attachable to this node
            for int_index in eval.exp.subspace_attachable_pairs[s_i]

                # Select an interaction for this arc
                eval.selected_pair_ints[node.arc_index] = int_index

                s_next, mat = eval.exp.pair_operator_mat[int_index][1][s_i]
                pos != 1 && pushfirst!(eval.matrix_prod, eval.ppgf_mats[pos - 1, s_i])
                pushfirst!(eval.matrix_prod, mat)

                _traverse_configuration_tree!(eval,
                                              pos + 1,
                                              s_next, s_f,
                                              pair_int_weight)

                popfirst!(eval.matrix_prod, pos == 1 ? 1 : 2)
            end

        else # Tail of an interaction arc

            int_index = eval.selected_pair_ints[node.arc_index]

            op_sbm = eval.exp.pair_operator_mat[int_index][2]
            if haskey(op_sbm, s_i)

                s_next, mat = op_sbm[s_i]
                pos != 1 && pushfirst!(eval.matrix_prod, eval.ppgf_mats[pos - 1, s_i])
                pushfirst!(eval.matrix_prod, mat)

                pair_int_weight_next =
                    eval.pair_ints[node.arc_index, int_index] * pair_int_weight

                _traverse_configuration_tree!(eval,
                                              pos + 1,
                                              s_next, s_f,
                                              pair_int_weight_next)

                popfirst!(eval.matrix_prod, pos == 1 ? 1 : 2)

            end

        end

    elseif node.kind == operator_flag

        op_sbm = eval.exp.corr_operators_mat[node.arc_index][node.operator_index]
        if haskey(op_sbm, s_i)
            s_next, op_mat = op_sbm[s_i]
            pos != 1 && pushfirst!(eval.matrix_prod, eval.ppgf_mats[pos - 1, s_i])
            pushfirst!(eval.matrix_prod, op_mat)

            _traverse_configuration_tree!(eval,
                                          pos + 1,
                                          s_next,
                                          s_f,
                                          pair_int_weight)

            popfirst!(eval.matrix_prod, pos == 1 ? 1 : 2)

        end

    elseif node.kind ∈ (identity_flag, inch_flag)

        pos != 1 && pushfirst!(eval.matrix_prod, eval.ppgf_mats[pos - 1, s_i])

        _traverse_configuration_tree!(eval,
                                      pos + 1,
                                      s_i, s_f,
                                      pair_int_weight)

        pos != 1 && popfirst!(eval.matrix_prod)

    else
        @assert false
    end

    return
end

end # module topology_eval
