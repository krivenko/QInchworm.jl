module topology_eval

using DocStringExtensions

using TimerOutputs: TimerOutput, @timeit

using Keldysh; kd = Keldysh

using QInchworm
using QInchworm; cfg = QInchworm.configuration

using QInchworm: SectorBlockMatrix
using QInchworm.expansion: Expansion
using QInchworm.configuration: Configuration,
                               Time,
                               InteractionEnum,
                               pair_flag,
                               identity_flag,
                               inch_flag,
                               operator_flag

using QInchworm.diagrammatics: Topology,
                               is_doubly_k_connected,
                               generate_topologies,
                               Diagram

using QInchworm.utility: LazyMatrixProduct, eval!

mutable struct Node
    "Interaction type of operator"
    kind::InteractionEnum
    "Index for pair interaction arc"
    arc_index::Int64
    "Index to operator"
    operator_index::Int64
end

struct FixedNode
    "Reference to operator"
    node::Node
    "Contour time point"
    time::kd.BranchPoint
end

"""
$(TYPEDSIGNATURES)

Returns a fixed node at time `time` with an associated identity operator.
"""
function IdentityNode(time::kd.BranchPoint)::FixedNode
    return FixedNode(Node(identity_flag, -1, -1), time)
end

"""
$(TYPEDSIGNATURES)

Returns a fixed "inch" node at time `time` with an associated identity operator.
"""
function InchNode(time::kd.BranchPoint)::FixedNode
    return FixedNode(Node(inch_flag, -1, -1), time)
end

"""
$(TYPEDSIGNATURES)

Returns a fixed operator node at time `time` with an associated operator.
"""
function OperatorNode(time::kd.BranchPoint,
                      operator_pair_index::Int64,
                      operator_index::Int64)::FixedNode
    return FixedNode(Node(operator_flag, operator_pair_index, operator_index), time)
end


"""
A list of topologies represented as a tree of arcs connecting various nodes in
configurations. All the topologies share the first arc, which is stored at the root of
the tree.
"""
struct ArcTree
    "Positions of head and tail nodes within a configuration"
    arc::Pair{Int, Int}
    "Index of the arc counting from the beginning of the configuration"
    arc_index::Int
    "Parity of a complete topology, relevant only at the leaves"
    parity::Int
    "Arc sub-trees"
    subtrees::Vector{ArcTree}
end

function _make_arc_trees(topologies::Vector{Topology},
                         top_to_conf_pos::Vector{Int})::Vector{ArcTree}
    arcs = Vector{Pair{Int, Int}}(undef, first(topologies).order)
    trees = Vector{ArcTree}()

    for top in topologies
        for (i, p) in enumerate(top.pairs)
            # Head and tail of a pair must be swapped here because of the reversed orders
            # of nodes in a topology and in a configuration.
            arcs[i] = top_to_conf_pos[p[2]] => top_to_conf_pos[p[1]]
        end
        # Sort the arcs by position of their head in the configuration
        sort!(arcs, by = arc -> arc[1])

        # Add branches to the tree
        s = trees
        for (arc_index, arc) in enumerate(arcs)
            a = findfirst(t -> t.arc == arc, s)
            if a === nothing
                # Parity is defined only at the leaves
                parity = (arc_index == top.order) ? top.parity : 0
                push!(s, ArcTree(arc, arc_index, parity, []))
                s = s[end].subtrees
            else
                s = s[a].subtrees
            end
        end
    end

    return trees
end

struct TopologyEvaluator

    "Pseudo-particle expansion problem"
    exp::Expansion

    "Expansion order"
    order::Int

    "Configuration as a list of nodes arranged in the contour order"
    conf::Vector{Node}

    "Contour positions of nodes in the configuration"
    times::Vector{kd.BranchPoint}

    "Correspondence of node positions within a topology and a configuration"
    top_to_conf_pos::Vector{Int64}

    "Arc trees"
    arc_trees::Vector{ArcTree}

    "Must the bold PPGFs be used?"
    use_bold_prop::Bool

    """
    PPGFs evaluated at all relevant pairs of time arguments.

    ppgf_mats[i, s] is the s-th diagonal block of exp.P (or exp.P0) evaluated at the pair
    of time points ``(t_{i+1}, t_i)``.
    """
    ppgf_mats::Array{Matrix{ComplexF64}, 2}

    """
    Pair interaction arcs evaluated at all relevant pairs of time arguments.

    pair_ints[n1, n2, p] is the propagator from `exp.pairs[p]` evaluated at the pair of time
    points corresponding to the configuration nodes n1 and n2 (n2 > n1).
    """
    pair_ints::Array{ComplexF64, 3}

    """
    Configuration position of the heads of the interaction arcs.
    """
    selected_head_pos::Vector{Int64}

    """
    Indices of pair interactions within `exp.pairs` assigned to each
    interaction arc in a topology.
    """
    selected_pair_ints::Vector{Int64}

    """Pre-allocated container for evaluation results"""
    result_mats::Vector{Matrix{ComplexF64}}

    """Pre-allocated matrix product evaluator"""
    matrix_prod::LazyMatrixProduct{ComplexF64}

    """Internal performance timer"""
    tmr::TimerOutput

    function TopologyEvaluator(exp::Expansion,
                               order::Int,
                               topologies::Vector{Topology},
                               fixed_nodes::Dict{Int, FixedNode};
                               tmr::TimerOutput = TimerOutput())
        n_nodes = 2 * order + length(fixed_nodes)
        @assert maximum(keys(fixed_nodes)) <= n_nodes

        # Prepare a skeleton of the configuration
        conf = [Node(pair_flag, 0, 1) for i in 1:n_nodes]
        times = Vector{kd.BranchPoint}(undef, n_nodes)

        use_bold_prop = false
        for (pos, fn) in fixed_nodes
            conf[pos] = fn.node

            if fn.node.kind == inch_flag || fn.node.kind == operator_flag
                use_bold_prop = true
            end

            # Copy time of fixed nodes to `times`
            times[pos] = fixed_nodes[pos].time
        end

        # Build the `top_to_conf_pos` map.
        # We need the reverse() here because the orders of nodes in a topology and in a
        # configuration are reversed.
        top_to_conf_pos = [pos for pos in reverse(1:n_nodes) if !haskey(fixed_nodes, pos)]

        # Build the arc tree
        arc_trees = _make_arc_trees(topologies, top_to_conf_pos)

        ppgf_mats = [Matrix{ComplexF64}(undef, norbitals(p), norbitals(p))
                     for _ in 1:(n_nodes-1), p in exp.P]
        pair_ints = Array{ComplexF64}(undef, n_nodes, n_nodes, length(exp.pairs))
        selected_head_pos = Vector{Int64}(undef, order)
        selected_pair_ints = Vector{Int64}(undef, order)
        result_mats = [zeros(ComplexF64, norbitals(p), norbitals(p)) for p in exp.P]
        matrix_prod = LazyMatrixProduct(ComplexF64, 2 * n_nodes - 1)

        return new(exp,
                   order,
                   conf,
                   times,
                   top_to_conf_pos,
                   arc_trees,
                   use_bold_prop,
                   ppgf_mats,
                   pair_ints,
                   selected_head_pos,
                   selected_pair_ints,
                   result_mats,
                   matrix_prod,
                   tmr)
    end
end

function (eval::TopologyEvaluator)(times::Vector{kd.BranchPoint})::SectorBlockMatrix
    @boundscheck length(times) == 2 * eval.order

    # Update eval.times
    for (pos, t) in zip(eval.top_to_conf_pos, times)
        eval.times[pos] = t
    end

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
                kd.interpolate!(eval.ppgf_mats[i, s], eval.exp.P[s], time_f, time_i)
            else
                kd.interpolate!(eval.ppgf_mats[i, s], eval.exp.P0[s], time_f, time_i)
            end
            eval.ppgf_mats[i, s] *= im
        end
    end

    result_mats = [zeros(ComplexF64, norbitals(p), norbitals(p)) for p in eval.exp.P]

    # Pre-compute eval.pair_ints
    for i1 = 1:(2 * eval.order), i2 = (i1 + 1):(2 * eval.order)
        pos_head = eval.top_to_conf_pos[i2]
        pos_tail = eval.top_to_conf_pos[i1]
        @assert pos_tail > pos_head

        time_i = eval.times[pos_head]
        time_f = eval.times[pos_tail]
        # Tackle time ordering violations caused by rounding errors
        if time_f < time_i
            time_f = time_i
        end

        for (p, int_pair) in enumerate(eval.exp.pairs)
            eval.pair_ints[pos_head, pos_tail, p] = im * int_pair.propagator(time_f, time_i)
        end
    end

    @timeit eval.tmr "Tree traversal" begin

    fill!.(eval.result_mats, 0.0)

    # Traverse the configuration tree for each initial subspace
    for s_i in eachindex(eval.exp.P) # TODO: Parallelization opportunity
        @assert eval.matrix_prod.n_mats == 0
        _traverse_configuration_tree!(eval,
                                      1,
                                      s_i, s_i,
                                      eval.arc_trees,
                                      ComplexF64(1))
    end

    end # tmr

    return Dict(s => (s, -im * (-1)^eval.order * mat)
                for (s, mat) in enumerate(eval.result_mats))
end

"""
Recursively traverse a tree of all configurations (lists of nodes) stemming from given
topology and contributing to the quantity of interest.

eval            : Evaluator object.
pos             : Position of the currently processed node in the configuration.
s_i             : Left block index of the matrix representation of the current node.
s_f             : Right block index expected at the final node.
arc_subtrees    : (Sub-)trees of pair interaction arcs.
pair_int_weight : Current weight of the pair interaction contribution.
"""
function _traverse_configuration_tree!(eval::TopologyEvaluator,
                                       pos::Int,
                                       s_i::Int,
                                       s_f::Int,
                                       arc_subtrees::Vector{ArcTree},
                                       pair_int_weight::ComplexF64)

    # Are we at a leaf?
    if pos > length(eval.conf)
        if s_i == s_f # Is the resulting configuration block-diagonal?
            eval.result_mats[s_i] .+= pair_int_weight * eval!(eval.matrix_prod)
        end
        return
    end

    node = eval.conf[pos] # Current node

    if node.kind == pair_flag

        if node.operator_index == 1 # Head of an interaction arc

            # Loop over all nodes that can serve as the tail for the arc starting here
            for arc_s in arc_subtrees

                head_pos, tail_pos = arc_s.arc
                eval.selected_head_pos[arc_s.arc_index] = head_pos

                # Update the tail node
                eval.conf[tail_pos].arc_index = arc_s.arc_index
                eval.conf[tail_pos].operator_index = 2

                # Are we at the head of the last arc? Then multiply the interaction weight
                # by the parity
                next_pair_int_weight = (arc_s.arc_index == eval.order ?
                                        arc_s.parity : 1) * pair_int_weight

                # Loop over all interaction pairs attachable to this node
                for int_index in eval.exp.subspace_attachable_pairs[s_i]

                    # Select an interaction for this arc
                    eval.selected_pair_ints[arc_s.arc_index] = int_index

                    s_next, mat = eval.exp.pair_operator_mat[int_index][1][s_i]
                    pos != 1 && pushfirst!(eval.matrix_prod, eval.ppgf_mats[pos - 1, s_i])
                    pushfirst!(eval.matrix_prod, mat)

                    _traverse_configuration_tree!(eval,
                                                  pos + 1,
                                                  s_next, s_f,
                                                  arc_s.subtrees,
                                                  next_pair_int_weight)

                    popfirst!(eval.matrix_prod, pos == 1 ? 1 : 2)

                end

                # Reset the tail node
                eval.conf[tail_pos].arc_index = 0
                eval.conf[tail_pos].operator_index = 1

            end

        else # Tail of an interaction arc

            int_index = eval.selected_pair_ints[node.arc_index]

            op_sbm = eval.exp.pair_operator_mat[int_index][2]

            head_pos = eval.selected_head_pos[node.arc_index]

            if haskey(op_sbm, s_i)

                s_next, mat = op_sbm[s_i]
                pos != 1 && pushfirst!(eval.matrix_prod, eval.ppgf_mats[pos - 1, s_i])
                pushfirst!(eval.matrix_prod, mat)

                pair_int_weight_next =
                    eval.pair_ints[head_pos, pos, int_index] * pair_int_weight

                _traverse_configuration_tree!(eval,
                                              pos + 1,
                                              s_next, s_f,
                                              arc_subtrees,
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
                                          arc_subtrees,
                                          pair_int_weight)

            popfirst!(eval.matrix_prod, pos == 1 ? 1 : 2)

        end

    elseif node.kind ∈ (identity_flag, inch_flag)

        pos != 1 && pushfirst!(eval.matrix_prod, eval.ppgf_mats[pos - 1, s_i])

        _traverse_configuration_tree!(eval,
                                      pos + 1,
                                      s_i, s_f,
                                      arc_subtrees,
                                      pair_int_weight)

        pos != 1 && popfirst!(eval.matrix_prod)

    else
        @assert false
    end

end

end # module topology_eval
