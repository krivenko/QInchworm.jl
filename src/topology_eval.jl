module topology_eval

using DocStringExtensions

using Keldysh; kd = Keldysh

using QInchworm; cfg = QInchworm.configuration

using QInchworm: SectorBlockMatrix
using QInchworm.expansion: Expansion
using QInchworm.configuration: Configuration,
                               Time,
                               Node,
                               InchNode,
                               NodePair,
                               is_inch_node,
                               identity_flag
using QInchworm.diagrammatics: Topology,
                               Diagram,
                               pair_partitions,
                               count_doubly_k_connected,
                               is_doubly_k_connected,
                               generate_topologies

using QInchworm.utility: rank_sub_range, mpi_all_gather_julia_vector


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
        pair_idx_range = range(1, length(expansion.pairs)) # range of allowed pp interaction pair indices
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

    diagrams_out = mpi_all_gather_julia_vector(rank_diagrams)
    
    if return_configurations
        configurations = mpi_all_gather_julia_vector(rank_configurations)
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

    diagrams_out = mpi_all_gather_julia_vector(rank_diagrams_out)
    if return_configurations
        configurations = mpi_all_gather_julia_vector(rank_configurations)
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
        configuration.nodes[n_idx] = Node(times[t_idx], op_ref)
    end

    for (p_idx, (idx_tf, idx_ti)) in enumerate(diagram.topology.pairs)
        int_idx = configuration.pairs[p_idx].index
        configuration.pairs[p_idx] = NodePair(times[idx_tf], times[idx_ti], int_idx)
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

end # module topology_eval
