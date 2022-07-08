module topology_eval

using DocStringExtensions

import Keldysh; kd = Keldysh

import QInchworm; cfg = QInchworm.configuration
import QInchworm; diag = QInchworm.diagrammatics

import QInchworm.configuration: Expansion
import QInchworm.configuration: Node, InchNode, NodePair, NodePairs


""" Get configuration.Nodes defining the inch-worm interval [τ_f, τ_w, τ_0]
    
Parameters
----------
grid : Keldysh.FullTimeGrid
fidx : Final imaginary time discretization index for the current inching setup
widx : Worm time in the current inching setup (usually widx = fidx - 1)

Returns
-------
worm_nodes : QInchworm.configuration.Nodes with the three times as configuration.Node
    
"""
function get_imaginary_time_worm_nodes(grid::kd.FullTimeGrid, fidx::Int64, widx::Int64)::cfg.Nodes

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

worm_nodes : QInchworm.configuration.Nodes containing [n_f, n_w, n_0] where 
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
    contour::kd.AbstractContour, worm_nodes::cfg.Nodes, x1::Float64, xs::Vector{Float64}
    )::Vector{kd.BranchPoint}
    
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


"""
Diagram with a topology and tuple of pseudo particle interaction pair indices
"""
struct Diagram
  "Topology"
  topology::diag.Topology
  "Pair indices"
  pair_idxs::Tuple{Vararg{Int64}}
end


"""
"""
const Diagrams = Vector{Diagram}


function get_topologies_at_order(order::Int64)::Vector{diag.Topology}

    topologies = diag.Topology.(diag.pair_partitions(order))
    
    k = 1
    filter!(topologies) do top
        diag.is_k_connected(top, k)
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
    expansion::cfg.Expansion, topologies::Vector{diag.Topology}, order::Int64
    )::Diagrams
    
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
    expansion::cfg.Expansion, worm_nodes::cfg.Nodes, τs::Vector{kd.BranchPoint}, diagrams::Diagrams
    )::cfg.SectorBlockMatrix

    accumulated_value = 0 * cfg.operator(expansion, first(worm_nodes))
    
    for (didx, diagram) in enumerate(diagrams)
        nodepairs = [ NodePair(τs[a], τs[b], diagram.pair_idxs[idx])
                      for (idx, (a, b)) in enumerate(diagram.topology.pairs) ]
        configuration = Configuration(worm_nodes, nodepairs)
        value = cfg.eval(expansion, configuration)
        accumulated_value += value
    end
    
    return accumulated_value
end

end # module topology_eval
