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

""" Get a list of `configuration.Node`'s defining the inchworm interval [τ_f, τ_w, τ_0]

Parameters
----------
grid : Keldysh.FullTimeGrid
fidx : Final imaginary time discretization index for the current inching setup
widx : Worm time in the current inching setup (usually widx = fidx - 1)

Returns
-------
worm_nodes : A list of the three times as `configuration.Node`'s

"""
function get_imaginary_time_worm_nodes(grid::kd.FullTimeGrid, fidx::Int64, widx::Int64)::Vector{Node}

    @assert fidx >= widx

    τ_grid = grid[kd.imaginary_branch]

    @assert fidx <= length(τ_grid)

    τ_0 = τ_grid[1]
    τ_f = τ_grid[fidx]
    τ_w = τ_grid[widx]

    n_f = Node(τ_f.bpoint)
    n_w = InchNode(τ_w.bpoint)
    n_0 = Node(τ_0.bpoint)

    @assert n_f.time.ref >= n_w.time.ref >= n_0.time.ref

    worm_nodes = [n_f, n_w, n_0]

    return worm_nodes
end


"""Helper function for mapping random unit-inteval random numbers to inch-worm imaginary contour times

Parameters
----------

worm_nodes : A list of `configuration.Node`'s containing [n_f, n_w, n_0] where
             - n_f is the final time node
             - n_w is the worm time node
             - n_0 is the initial time node (τ=0)
             NB! Can be generated with `get_imaginary_time_worm_nodes`
x1    : unit interval number x1 ∈ [0, 1] mapped to τ_1 ∈ [τ_f, τ_w]
xs    : *reverse* sorted list of unit-interval numbers
         xs[i] ∈ [0, 1] mapped to τ[i] ∈ [τ_w, τ_0]

Returns
-------
τs : Vector of contour times with τ_1 as first element

"""
function timeordered_unit_interval_points_to_imaginary_branch_inch_worm_times(
    contour::kd.AbstractContour, worm_nodes::Vector{Node}, x1::Float64, xs::Vector{Float64}
    )::Vector{Time}

    x_f, x_w, x_0 = [ node.time.ref for node in worm_nodes ]

    # -- Scale first time x1 to the range [x_w, x_f]
    x1 = x_w .+ (x_f - x_w) * x1

    # -- Scale the other times to the range [x_w, x_0]
    xs = x_0 .+ (x_w - x_0) * xs

    xs = vcat([x1], xs) # append x1 first in the list

    # -- Transform xs to contour Keldysh.BranchPoint's
    τ_branch = contour[kd.imaginary_branch]
    τs = [ kd.get_point(τ_branch, x) for x in xs ]

    return τs
end

function get_topologies_at_order(order::Int64, k = nothing; with_1k_arc = false)::Vector{Topology}

    if with_1k_arc
        topologies = Topology.(pair_partitions(order), k = k)
    else
      topologies = generate_topologies(order)
#        topologies = Topology.(pair_partitions(order))
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
worm_nodes : Generated with `get_imaginary_time_worm_nodes`
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

    accumulated_value = zeros(SectorBlockMatrix, expansion.ed)

    for (didx, diagram) in enumerate(diagrams)
        nodepairs = [ NodePair(τs[f], τs[i], diagram.pair_idxs[idx])
                      for (idx, (f, i)) in enumerate(diagram.topology.pairs) ]
        configuration = Configuration(worm_nodes, nodepairs, expansion)
        value = cfg.eval(expansion, configuration)
        accumulated_value += value
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
    τ_2::kd.BranchPoint,
    τ_f::kd.BranchPoint)
    @assert configuration.op_node_idx !== nothing

    op_ref = configuration.nodes[configuration.op_node_idx[1]].operator_ref
    configuration.nodes[configuration.op_node_idx[1]] = Node(τ_1, op_ref)
    op_ref = configuration.nodes[configuration.op_node_idx[2]].operator_ref
    configuration.nodes[configuration.op_node_idx[2]] = Node(τ_2, op_ref)
    configuration.nodes[end] = Node(τ_f)
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
