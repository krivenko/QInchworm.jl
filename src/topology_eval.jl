module topology_eval

using DocStringExtensions
using LinearAlgebra

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


function get_topologies_at_order(order::Int64, k = nothing; with_external_arc = false)::Vector{Topology}
    topologies = generate_topologies(order)
    k === nothing && return topologies

    filter!(topologies) do top
        is_doubly_k_connected(top, k)
    end

    if with_external_arc
        topologies = [Topology(top.pairs, (-1)^k * top.parity) for top in topologies]
    end

    return topologies
end

""" Get all diagrams as combinations of a `Topology` and a list of pseudo particle interaction indicies

Parameters
----------

expansion : Pseudo particle expansion
order     : Inch worm perturbation order

Returns
-------

diagrams : Vector with tuples of topologies and pseudo particle interaction indicies

"""
function get_diagrams_at_order(
    expansion::Expansion, topologies::Vector{Topology}, order::Int64
    )::Vector{Diagram}

    # -- Generate all `order` lenght vector of combinations of pseudo particle interaction pair indices
    pair_idx_range = 1:length(expansion.pairs) # range of allowed pp interaction pair indices
    pair_idxs_combinations = collect(Iterators.product(repeat([pair_idx_range], outer=[order])...))

    diagrams = vec([ Diagram(topology, pair_idxs) for (topology, pair_idxs) in
            collect(Iterators.product(topologies, pair_idxs_combinations)) ])

    return diagrams
end

function get_configurations_and_diagrams_from_topologies(
    expansion::Expansion,
    #
    topologies::Vector{Topology}, order::Int64,
    #
    d_before::Union{Int, Nothing};
    op_pair_idx::Union{Int, Nothing} = nothing,
    return_configurations = true)::Tuple{Vector{Configuration}, Vector{Diagram}}

    r = rank_sub_range(length(topologies))
    rank_topologies = topologies[r]

    rank_diagrams = Diagram[]
    rank_configurations = Configuration[]

    for topology in rank_topologies

        # -- Generate all `order` lenght vector of combinations of pseudo particle interaction pair indices
        pair_idx_range = 1:length(expansion.pairs) # range of allowed pp interaction pair indices
        pair_idxs_combinations = Iterators.product(repeat([pair_idx_range], outer=[order])...)

        for pair_idxs in pair_idxs_combinations
            diagram = Diagram(topology, pair_idxs)

            if op_pair_idx === nothing
                configuration = Configuration(diagram, expansion, d_before)
            else
                configuration = Configuration(diagram, expansion, d_before, op_pair_idx)
            end

            if length(configuration.paths) > 0
                if return_configurations
                    push!(rank_configurations, configuration)
                end
                push!(rank_diagrams, diagram)
            end
        end
    end

    diagrams_out = all_gather(rank_diagrams)

    if return_configurations
        configurations = all_gather(rank_configurations)
    end

    if return_configurations
        return configurations, diagrams_out
    else
        return [], diagrams_out
    end
end

function get_configurations_and_diagrams_serial(
    expansion::Expansion,
    diagrams::Vector{Diagram},
    d_before::Union{Int, Nothing};
    op_pair_idx::Union{Int, Nothing} = nothing)::Tuple{Vector{Configuration}, Vector{Diagram}}

    diagrams_out = Diagram[]
    configurations = Configuration[]
    for (didx, diagram) in enumerate(diagrams)
        if op_pair_idx === nothing
            configuration = Configuration(diagram, expansion, d_before)
        else
            configuration = Configuration(diagram, expansion, d_before, op_pair_idx)
        end

        if length(configuration.paths) > 0
            push!(configurations, configuration)
            push!(diagrams_out, diagram)
        end
    end
    return configurations, diagrams_out
end

function get_configurations_and_diagrams(
    expansion::Expansion,
    diagrams::Vector{Diagram},
    d_before::Union{Int, Nothing};
    op_pair_idx::Union{Int, Nothing} = nothing,
    return_configurations = true)::Tuple{Vector{Configuration}, Vector{Diagram}}

    r = rank_sub_range(length(diagrams))
    rank_diagrams = diagrams[r]

    rank_diagrams_out = Diagram[]
    rank_configurations = Configuration[]
    for (didx, diagram) in enumerate(rank_diagrams)
        if op_pair_idx === nothing
            configuration = Configuration(diagram, expansion, d_before)
        else
            configuration = Configuration(diagram, expansion, d_before, op_pair_idx)
        end

        if length(configuration.paths) > 0
            if return_configurations
                push!(rank_configurations, configuration)
            end
            push!(rank_diagrams_out, diagram)
        end
    end

    diagrams_out = all_gather(rank_diagrams_out)
    if return_configurations
        configurations = all_gather(rank_configurations)
    end

    if return_configurations
        return configurations, diagrams_out
    else
        return [], diagrams_out
    end
end

function update_pair_node_times!(configuration::Configuration, diagram::Diagram, times::Vector{Time})
    for (t_idx, n_idx) in enumerate(configuration.pair_node_idxs)
        op_ref = configuration.nodes[n_idx].operator_ref
        configuration.nodes[n_idx] = cfg.Node(times[t_idx], op_ref)
    end

    for (p_idx, (idx_tf, idx_ti)) in enumerate(diagram.topology.pairs)
        int_idx = configuration.pairs[p_idx].index
        configuration.pairs[p_idx] = cfg.NodePair(times[idx_tf], times[idx_ti], int_idx)
    end
end

function eval(expansion::Expansion,
              diagrams::Vector{Diagram},
              configurations::Vector{Configuration},
              times::Vector{kd.BranchPoint},
    )::SectorBlockMatrix

    value = zeros(SectorBlockMatrix, expansion.ed)

    for (diagram, configuration) in zip(diagrams, configurations)
        update_pair_node_times!(configuration, diagram, times)
        cfg.eval_acc!(value, expansion, configuration)
    end

    return value
end

struct Node
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

struct TopologyEvaluator
    "Pseudo-particle expansion problem"
    exp::Expansion

    "Configuration as a list of nodes arranged in the contour order"
    conf::Vector{Node}

    "Contour positions of nodes in the configuration"
    times::Vector{kd.BranchPoint}

    "Correspondence of node positions within a topology and a configuration"
    top_to_conf_pos::Vector{Int64}

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

    pair_ints[a, p] is the propagator from `exp.pairs[p]` evaluated at the pair of time
    points corresponding to the a-th arc in a topology.
    """
    pair_ints::Array{ComplexF64, 2}

    """
    Indices of pair interactions within `exp.pairs` assigned to each
    interaction arc in a topology.
    """
    selected_pair_ints::Vector{Int64}

    """Pre-allocated container for per-topology evaluation results"""
    top_result::SectorBlockMatrix

    function TopologyEvaluator(exp::Expansion,
                               order::Int,
                               fixed_nodes::Dict{Int, FixedNode})
        n_nodes = 2 * order + length(fixed_nodes)
        @assert maximum(keys(fixed_nodes)) <= n_nodes

        # Prepare a skeleton of the configuration by placing only the fixed nodes
        conf = Vector{Node}(undef, n_nodes)
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
        # configurations are reversed.
        top_to_conf_pos = [pos for pos in reverse(1:n_nodes) if !haskey(fixed_nodes, pos)]

        ppgf_mats = [Matrix{ComplexF64}(undef, norbitals(p), norbitals(p))
                     for _ in 1:(n_nodes-1), p in exp.P]
        pair_ints = Array{ComplexF64}(undef, order, length(exp.pairs))

        selected_pair_ints = Vector{Int64}(undef, order)

        return new(exp,
                   conf,
                   times,
                   top_to_conf_pos,
                   use_bold_prop,
                   ppgf_mats,
                   pair_ints,
                   selected_pair_ints,
                   zeros(SectorBlockMatrix, exp.ed))
    end
end

function (eval::TopologyEvaluator)(topology::Topology,
    times::Vector{kd.BranchPoint})::SectorBlockMatrix
    return eval([topology], times)
end

function (eval::TopologyEvaluator)(topologies::Vector{Topology},
                                   times::Vector{kd.BranchPoint})::SectorBlockMatrix

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
                eval.ppgf_mats[i, s] = eval.exp.P[s](time_f, time_i)
            else
                eval.ppgf_mats[i, s] = eval.exp.P0[s](time_f, time_i)
            end
        end
    end

    result = zeros(SectorBlockMatrix, eval.exp.ed)

    for top in topologies # TODO: Parallelization opportunity I

        @assert length(times) == 2 * length(top.pairs)

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
                eval.pair_ints[a, p] = int_pair.propagator(time_f, time_i)
            end
        end

        fill!(eval.top_result, 0.0)

        # Traverse the configuration tree for each initial subspace
        for s_i in eachindex(eval.exp.P) # TODO: Parallelization opportunity II
            _traverse_configuration_tree!(eval,
                                          view(eval.conf, :),
                                          s_i, s_i,
                                          eval.exp.identity_mat[s_i][2],
                                          ComplexF64(1))
        end

        result += -im * top.parity * (-1)^top.order * eval.top_result
    end

    return result
end

function _traverse_configuration_tree!(eval::TopologyEvaluator,
                                       conf,
                                       s_i::Int64,
                                       s_f::Int64,
                                       ppgf_weight::Matrix{ComplexF64},
                                       pair_int_weight::ComplexF64)

    # Are we at a leaf?
    if isempty(conf)
        if s_i == s_f # Is the resulting configuration block-diagonal?
            val = pair_int_weight * ppgf_weight
            eval.top_result[s_i] = (s_f, eval.top_result[s_i][2] + val)
        end
        return
    end

    # Current position within the configuration
    pos = length(parent(conf)) - length(conf) + 1

    node = conf[1]                  # Current node
    conf_tail = @view conf[2:end]   # The rest of the configuration

    if node.kind == pair_flag

        if node.operator_index == 1 # Head of an interaction arc

            # Loop over all interaction pairs attachable to this node
            for int_index in eval.exp.subspace_attachable_pairs[s_i]

                # Select an interaction for this arc
                eval.selected_pair_ints[node.arc_index] = int_index

                s_next, mat = eval.exp.pair_operator_mat[int_index][1][s_i]
                ppgf_weight_next = (pos == 1) ?
                                   mat * ppgf_weight :
                                   mat * (im * eval.ppgf_mats[pos - 1, s_i]) * ppgf_weight

                _traverse_configuration_tree!(eval,
                                              conf_tail,
                                              s_next, s_f,
                                              ppgf_weight_next,
                                              pair_int_weight)
            end

        else # Tail of an interaction arc

            int_index = eval.selected_pair_ints[node.arc_index]

            op_sbm = eval.exp.pair_operator_mat[int_index][2]
            if haskey(op_sbm, s_i)

                s_next, mat = op_sbm[s_i]
                ppgf_weight_next = (pos == 1) ?
                                   mat * ppgf_weight :
                                   mat * (im * eval.ppgf_mats[pos - 1, s_i]) * ppgf_weight

                pair_int_weight_next =
                    im * eval.pair_ints[node.arc_index, int_index] * pair_int_weight

                _traverse_configuration_tree!(eval,
                                              conf_tail,
                                              s_next, s_f,
                                              ppgf_weight_next,
                                              pair_int_weight_next)
            end

        end

    elseif node.kind == operator_flag

        op_sbm = eval.exp.corr_operators_mat[node.arc_index][node.operator_index]
        if haskey(op_sbm, s_i)
            s_next, op_mat = op_sbm[s_i]
            ppgf_weight_next = (pos == 1) ?
                               op_mat * ppgf_weight :
                               op_mat * (im * eval.ppgf_mats[pos - 1, s_i]) * ppgf_weight
            _traverse_configuration_tree!(eval,
                                          conf_tail,
                                          s_next,
                                          s_f,
                                          ppgf_weight_next,
                                          pair_int_weight)
        end

    elseif node.kind ∈ (identity_flag, inch_flag)

        ppgf_weight_next = (pos == 1) ?
                           ppgf_weight :
                           (im * eval.ppgf_mats[pos - 1, s_i]) * ppgf_weight
        _traverse_configuration_tree!(eval,
                                      conf_tail,
                                      s_i, s_f,
                                      ppgf_weight_next,
                                      pair_int_weight)

    else
        @assert false
    end

end

end # module topology_eval
