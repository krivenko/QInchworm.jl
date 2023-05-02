module topology_eval

using DocStringExtensions

using Keldysh; kd = Keldysh

using QInchworm; cfg = QInchworm.configuration

using QInchworm: SectorBlockMatrix
using QInchworm.expansion: Expansion
using QInchworm.configuration: Configuration, Time, Node, InchNode, NodePair
using QInchworm.diagrammatics: Topology,
                               Diagram,
                               pair_partitions,
                               count_doubly_k_connected,
                               is_doubly_k_connected,
                               generate_topologies


function get_topologies_at_order(order::Int64, k = nothing; with_1k_arc = false)::Vector{Topology}
    topologies = generate_topologies(order)

    if with_1k_arc
      topologies = [Topology(top.pairs, (-1)^k * top.parity) for top in topologies]
    end
    k === nothing && return topologies

    filter!(topologies) do top
        is_doubly_k_connected(top, k)
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
    pair_idx_range = range(1, length(expansion.pairs)) # range of allowed pp interaction pair indices
    pair_idxs_combinations = collect(Iterators.product(repeat([pair_idx_range], outer=[order])...))

    diagrams = vec([ Diagram(topology, pair_idxs) for (topology, pair_idxs) in
            collect(Iterators.product(topologies, pair_idxs_combinations)) ])

    return diagrams
end


"""Evaluate all diagrams for a given set of internal times `τs` and external `worm_nodes` (worm times)

Parameters
----------

expansion  : Pseudo particle expansion
worm_nodes : List of nodes defining the inchworm interval [τ_0, τ_w, τ_f]
τs         : Internal diagram times generated with
             `timeordered_unit_interval_points_to_imaginary_branch_inch_worm_times`
diagrams   : All diagrams as combinations of one Topology and a tuple of pseudo particle interaction indices

Returns
-------

accumulated_value : Of all diagrams

"""
function eval(
    expansion::Expansion, worm_nodes::Vector{Node}, τs::Vector{kd.BranchPoint}, diagrams::Vector{Diagram}
    )::SectorBlockMatrix

    accumulated_value = SectorBlockMatrix()

    for (didx, diagram) in enumerate(diagrams)
        nodepairs = [ NodePair(τs[f], τs[i], diagram.pair_idxs[idx])
                      for (idx, (f, i)) in enumerate(diagram.topology.pairs) ]
        configuration = Configuration(worm_nodes, nodepairs, expansion)
        accumulated_value += cfg.eval(expansion, configuration)
    end

    return accumulated_value
end

function get_configurations_and_diagrams(expansion::Expansion,
                                         diagrams::Vector{Diagram},
                                         d_before::Int;
                                         op_pair_idx::Union{Int, Nothing} = nothing)::Tuple{Vector{Configuration}, Vector{Diagram}}
    diagrams_out = Diagram[]
    configurations = Configuration[]
    for (didx, diagram) in enumerate(diagrams)
        configuration = op_pair_idx === nothing ? Configuration(diagram, expansion, d_before) :
                                                  Configuration(diagram, expansion, d_before, op_pair_idx)
        if length(configuration.paths) > 0
            push!(configurations, configuration)
            push!(diagrams_out, diagram)
        end
    end
    return configurations, diagrams_out
end

function update_inch_times!(configuration::Configuration, τ_i::kd.BranchPoint, τ_w::kd.BranchPoint, τ_f::kd.BranchPoint)
    if configuration.has_inch_node
        @inbounds begin
            configuration.nodes[1] = Node(τ_i)
            configuration.nodes[configuration.inch_node_idx] = InchNode(τ_w)
            configuration.nodes[end] = Node(τ_f)
        end
    else
        @inbounds begin
            configuration.nodes[1] = Node(τ_i)
            configuration.nodes[end] = Node(τ_f)
        end
    end
end

function update_times!(configuration::Configuration, diagram::Diagram, times::Vector{Time})
    for (t_idx, n_idx) in enumerate(configuration.node_idxs)
        op_ref = configuration.nodes[n_idx].operator_ref
        configuration.nodes[n_idx] = Node(times[t_idx], op_ref)
    end

    for (p_idx, (idx_tf, idx_ti)) in enumerate(diagram.topology.pairs)
        int_idx = configuration.pairs[p_idx].index
        configuration.pairs[p_idx] = NodePair(times[idx_tf], times[idx_ti], int_idx)
    end
end

# TODO: Should this function be merged with update_inch_times!() ?
function update_corr_times!(configuration::Configuration,
    τ_1::kd.BranchPoint,
    τ_2::kd.BranchPoint)
    @assert configuration.op_node_idx !== nothing

    op_ref = configuration.nodes[configuration.op_node_idx[1]].operator_ref
    configuration.nodes[configuration.op_node_idx[1]] = Node(τ_1, op_ref)
    op_ref = configuration.nodes[configuration.op_node_idx[2]].operator_ref
    configuration.nodes[configuration.op_node_idx[2]] = Node(τ_2, op_ref)
end

function eval(expansion::Expansion,
              diagrams::Vector{Diagram},
              configurations::Vector{Configuration},
              times::Vector{kd.BranchPoint},
    )::SectorBlockMatrix

    value = zeros(SectorBlockMatrix, expansion.ed)

    for (diagram, configuration) in zip(diagrams, configurations)
        update_times!(configuration, diagram, times)
        cfg.eval_acc!(value, expansion, configuration)
    end

    return value
end

end # module topology_eval
