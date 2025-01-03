# QInchworm.jl
#
# Copyright (C) 2021-2025 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Authors: Joseph Kleinhenz, Igor Krivenko, Hugo U. R. Strand

"""
Strong coupling expansion diagrams and their topological properties.
"""
module diagrammatics

using DocStringExtensions

using Combinatorics: levicivita, doublefactorial

const PairVector = Vector{Pair{Int,Int}}

"""
    $(TYPEDEF)

Datatype for strong coupling diagram topology.

A topology of order ``n`` consists of a partition of the tuple ``s = \\{1,...,2n\\}``
into ``n`` pairs ``\\{(\\pi(1), \\pi(2)), ..., (\\pi(2n-1), \\pi(2n))\\}`` where ``\\pi`` is
a permutation of ``s``.
Diagrammatically, a topology can be thought of as a set of arcs connecting vertices located
at ``\\{1,...,2n\\}``.
The parity of the topology is the sign of the permutation ``\\pi``.

# Fields
$(TYPEDFIELDS)
"""
struct Topology
    "Topology order ``n``"
    order::Int
    "List of pairs ``\\{(\\pi(1), \\pi(2)), ..., (\\pi(2n-1), \\pi(2n))\\}``"
    pairs::PairVector
    "Parity of the permutation ``\\pi``"
    parity::Int

    function Topology(pairs::PairVector, parity::Int)
        return new(length(pairs), pairs, parity)
    end
end

"""
    $(TYPEDSIGNATURES)

Construct topology from a list of pairs.
"""
function Topology(pairs::PairVector)
    p = levicivita(collect(Iterators.flatten(pairs)))
    return Topology(pairs, p)
end

"""
    $(TYPEDSIGNATURES)

Check that topology is properly constructed, i.e. that its list of pairs is derived from
a permutation of `1:2order` and that the parity of the permutation is correct.
"""
function Base.isvalid(t::Topology)
    perm = collect(Iterators.flatten(t.pairs))
    return ((t.parity == levicivita(perm)) && (sort!(perm) == 1:(2*t.order)))
end

"""
    $(TYPEDSIGNATURES)

Given a pair, return its version with the smallest element coming first.
"""
function sortpair(p::Pair{T, T}) where {T}
    return p.first > p.second ? p.second => p.first : p
end

"""
    $(TYPEDSIGNATURES)

Return true if a given integer pair has one element <= `k` and the other element > `k`.
"""
function is_doubly_k_connected(p::Pair{Int, Int}, k::Int)
    return (p.first <= k && p.second > k) || (p.second <= k && p.first > k)
end

"""
    $(TYPEDSIGNATURES)

Return `true` if two arcs cross.

Let ``p_1 = (a_1, b_1)``, ``p_2 = (a_2, b_2)`` represent two arcs, where without loss of
generality we assume ``a_1 < b_1`` and ``a_2 < b_2``.
Now consider the order of the points ``\\{a_1, b_1, a_2, b_2\\}``.
The orderings ``a_1 b_1 a_2 b_2``, ``a_1 a_2 b_2 b_1``, ``a_2 b_2 a_1 b_1`` are all
non-crossing while ``a_1 a_2 b_1 b_2`` and ``a_2 a_1 b_2 b_1`` are crossing.
"""
function iscrossing(p1::Pair{Int, Int}, p2::Pair{Int, Int})
    p1 = sortpair(p1)
    p2 = sortpair(p2)
    (p1.first < p2.first && p2.first < p1.second && p1.second < p2.second) && return true
    (p2.first < p1.first && p1.first < p2.second && p2.second < p1.second) && return true
    return false
end

"""
    $(TYPEDSIGNATURES)

Return the number of crossing arcs in a topology.
"""
function n_crossings(top::Topology)::Int
    n = 0
    for i1 in 1:top.order, i2 in (i1 + 1):top.order # Iterate over all unique pairs of arcs
        if iscrossing(top.pairs[i1], top.pairs[i2])
            n += 1
        end
    end
    return n
end

"""
    $(TYPEDSIGNATURES)

Returns the parity of the topolgy.
"""
function parity(top::Topology)::Int
    return top.parity
end

"""
    $(TYPEDSIGNATURES)

Given a vector of `connected` arcs and a vector of `disconnected` arcs
recursively add disconnected to connected if they cross with any connected.
This is done by traversing the crossing graph using depth first search.
"""
function traverse_crossing_graph_dfs!(connected::PairVector, disconnected::PairVector)
    stack = copy(connected)
    resize!(connected, 0)
    while !isempty(stack)
        p_c = pop!(stack)
        push!(connected, p_c)
        # pairs which cross with p_c are connected
            for (i, p_dc) in Iterators.reverse(enumerate(disconnected))
            iscrossing(p_c, p_dc) && push!(stack, popat!(disconnected, i))
        end
    end
end

"""
    $(TYPEDSIGNATURES)

Given a vector of integer pairs, split it into a `connected` set containing pairs with
an element <= `k` and a disconnected set containing the rest.
"""
function split_k_connected(pairs::PairVector, k::Int)
    connected = PairVector()
    disconnected = PairVector()

    for p in pairs
        if p.first <= k || p.second <= k
            push!(connected, p)
        else
            push!(disconnected, p)
        end
    end

    return connected, disconnected
end

"""
    $(TYPEDSIGNATURES)

Given a vector of integer pairs, count the doubly `k`-connected ones.
"""
function count_doubly_k_connected(pairs::PairVector, k::Int)
    return count(p -> is_doubly_k_connected(p, k), pairs)
end

"""
    $(TYPEDSIGNATURES)

Given a vector of integer pairs, split it into a `connected` set containing pairs with
one element <= `k` and the other element > `k`, and a disconnected set containing the rest.
"""
function split_doubly_k_connected(pairs::PairVector, k::Int)
    connected = PairVector()
    disconnected = PairVector()

    for p in pairs
        if is_doubly_k_connected(p, k)
            push!(connected, p)
        else
            push!(disconnected, p)
        end
    end

    return connected, disconnected
end

"""
    $(TYPEDSIGNATURES)

Given a topology, check if every connected component of the graph induced by
crossings between the arcs contains a pair with an element <= `k`.
"""
function is_k_connected(t::Topology, k::Int)
    connected, disconnected = split_k_connected(t.pairs, k)
    traverse_crossing_graph_dfs!(connected, disconnected)
    return isempty(disconnected)
end

"""
    $(TYPEDSIGNATURES)

Given a topology, check if every connected component of the graph induced by
crossings between the arcs contains a pair with one element <= `k` and the other
element > `k`.
"""
function is_doubly_k_connected(t::Topology, k::Int)
    connected, disconnected = split_doubly_k_connected(t.pairs, k)
    traverse_crossing_graph_dfs!(connected, disconnected)
    return isempty(disconnected)
end


"""
    $(TYPEDSIGNATURES)

Return the pair `v[i] => v[j]` and a copy of the vector `v` with elements `i`, `j` removed.
"""
function pop_pair(v::Vector, i, j)
    return v[i] => v[j], [v[k] for k in eachindex(v) if k ∉ (i, j)]
end

"""
    $(TYPEDSIGNATURES)

Given a vector of pairs representing a partial partition of the vertices and a
vector of unpaired vertices, return a vector of complete partitions.
"""
function pair_partitions(pairs::PairVector, unpaired::Vector{Int})
    length(unpaired) == 0 && return [pairs]
    @assert length(unpaired) % 2 == 0
    return map(2:length(unpaired)) do i
        p, rest = pop_pair(unpaired, 1, i)
        return pair_partitions(push!(copy(pairs), p), rest)
    end |> Iterators.flatten |> collect
end

"""
    $(TYPEDSIGNATURES)

Return all possible partitions of an even number of vertices into pairs.
"""
function pair_partitions(vertices::Vector{Int})
    @assert length(vertices) % 2 == 0
    pairs = PairVector()
    return pair_partitions(pairs, vertices)
end

"""
    $(TYPEDSIGNATURES)

Return all partitions of ``\\{1,\\ldots,2n\\}`` into ``n`` pairs.
"""
function pair_partitions(n::Int)
    return pair_partitions(collect(1:(2n)))
end

"""
    $(TYPEDSIGNATURES)

Given a partial topology and a vector of unpaired vertices, return a vector of
complete topologies, efficiently computing the permutation sign.
"""
function generate_topologies_impl(topology_partial::Topology, unpaired::Vector{Int})
    length(unpaired) == 0 && return [topology_partial]
    @assert length(unpaired) % 2 == 0
    return map(2:length(unpaired)) do i
        p, rest = pop_pair(unpaired, 1, i)
        # NOTE: each step adds one swap => flips parity
        parity = topology_partial.parity * (-1)^(i)
        return generate_topologies_impl(
            Topology(push!(copy(topology_partial.pairs), p), parity),
            rest,
        )
    end |>
    Iterators.flatten |>
    collect
end

"""
    $(TYPEDSIGNATURES)

Return topologies of order ``n``, efficiently computing the permutation sign for each.
"""
function generate_topologies(n::Int)
    empty_top = Topology(PairVector(), 1)
    return generate_topologies_impl(empty_top, collect(1:(2*n)))
end

"""
    $(TYPEDSIGNATURES)

Return topologies of a given order. If `k` is specified, then only the doubly k-connected
topologies are returned. With `with_external_arc` set to `true` parity of the returned
topologies is computed as if an extra arc overarching the last `2order - k` vertices
was present.
"""
function get_topologies_at_order(order::Int,
                                 k = nothing;
                                 with_external_arc = false)::Vector{Topology}
    topologies = generate_topologies(order)
    isnothing(k) && return topologies

    filter!(topologies) do top
        is_doubly_k_connected(top, k)
    end

    if with_external_arc
        topologies = [Topology(top.pairs, (-1)^k * top.parity) for top in topologies]
    end

    return topologies
end

end # module diagrammatics
