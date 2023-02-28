module diagrammatics

using DocStringExtensions

using LinearAlgebra: det

const PairVector = Vector{Pair{Int,Int}}

# TODO implement a fast calculation here
"""
$(TYPEDSIGNATURES)

Returns sign of the permutation ``x``

Sign is computed by calculating the determinant of the permutation matrix.
Does not check if ``x`` is a valid permutation.
"""
function parity_slow(x::Vector{Int})
  n = length(x)

  P = zeros(Int, n, n)

  for (i, j) in enumerate(x)
    P[i, j] = 1
  end

  return Int(det(P))
end

"""
$(TYPEDEF)

Datatype for diagram topology.
A topology of order ``n`` consists of a partition of the ordered set ``s = \\{1,...,2n\\}``
into ``n`` pairs `\\{(x(1), x(2)), ..., (x(2n-1), x(2n))\\}` where ``x`` is a permutation of ``s``.
Diagrammatically a topology can be thought of as a set of arcs connecting vertices located at ``\\{1,...,2n\\}``.
The parity of the topology is the sign of the permutation ``x``.

$(TYPEDFIELDS)
"""
struct Topology
  order::Int
  pairs::PairVector
  parity::Int

  function Topology(pairs::PairVector, parity::Int)
    new(length(pairs), pairs, parity)
  end

end

function Topology(pairs::PairVector)
  p = parity_slow(collect(Iterators.flatten(pairs)))
  return Topology(pairs, p)
end


function Base.isvalid(t::Topology)
  perm = collect(Iterators.flatten(t.pairs))
  return ((t.parity == parity_slow(perm)) && (sort!(perm) == 1:(2*t.order)))
end

function sortpair(p::Pair{T,T}) where {T}
  return p.first > p.second ? p.second => p.first : p
end


"""
$(TYPEDSIGNATURES)

Returns true if two arcs cross

Let ``p_1 = (a, b)``, ``p_2 = (x, y)`` represent two arcs, where without loss of generality we assume ``a < b`` and ``x < y``.
Now consider the order the points ``\\{a, b, x, y\\}``.
The orderings ``abxy``, ``axyb``, ``xyab`` are all non-crossing while ``axby`` and ``xayb`` are crossing.
"""
function iscrossing(p1::Pair{Int,Int}, p2::Pair{Int,Int})
  p1 = sortpair(p1)
  p2 = sortpair(p2)
  (p1.first < p2.first && p2.first < p1.second && p1.second < p2.second) && return true
  (p2.first < p1.first && p1.first < p2.second && p2.second < p1.second) && return true
  return false
end

"""
$(TYPEDSIGNATURES)

Returns the number of crossing arcs in a topology.

"""
function n_crossings(top::Topology)::Int
  n = 0
  for i1 in range(1, top.order), i2 in range(i1 + 1, top.order) # Iterate over all unique pairs of arcs
    if iscrossing(top.pairs[i1], top.pairs[i2])
      n += 1
    end
  end
  return n
end

"""
$(TYPEDSIGNATURES)

Returns the parity of the permutation matrix of the topolgy.

"""
function parity(top::Topology)::Int
  return top.parity
end

"""
$(TYPEDSIGNATURES)

Given a vector of 'connected' arcs and a vector of 'disconnected' arcs
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

Given a vector of pairs, split it into a 'connected' set containing pairs with
an index <= `k` and a disconnected set containing the rest
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

Given a vector of pairs, split it into a 'connected' set containing pairs with
one index <= `k` and the other index > `k` and a disconnected set containing the rest
"""
function split_doubly_k_connected(pairs::PairVector, k::Int)
  connected = PairVector()
  disconnected = PairVector()

  for p in pairs
    if (p.first <= k && p.second > k) || (p.first > k && p.second <= k)
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
crossings between the arcs contains a pair with index <= `k`
"""
function is_k_connected(t::Topology, k::Int)
  connected, disconnected = split_k_connected(t.pairs, k)
  traverse_crossing_graph_dfs!(connected, disconnected)
  return isempty(disconnected)
end

"""
$(TYPEDSIGNATURES)

Given a topology, check if every connected component of the graph induced by
crossings between the arcs contains a pair with one index <= `k` and the other
index > `k`
"""
function is_doubly_k_connected(t::Topology, k::Int)
  connected, disconnected = split_doubly_k_connected(t.pairs, k)
  traverse_crossing_graph_dfs!(connected, disconnected)
  return isempty(disconnected)
end


"""
$(TYPEDSIGNATURES)

returns the pair `v[i] => v[j]` and a copy of the vector `v` with elements `i`,`j` removed
"""
function pop_pair(v::Vector, i, j)
  v[i] => v[j], [v[k] for k in eachindex(v) if k ∉ (i, j)]
end

"""
$(TYPEDSIGNATURES)

Given a vector of pairs representing a partial partition of the vertices and a
vector of unpaired vertices, return a vector of complete partitions.
"""
function pair_partitions(pairs::PairVector, unpaired::Vector{Int})
  length(unpaired) == 0 && return [pairs]
  @assert length(unpaired) % 2 == 0
  map(2:length(unpaired)) do i
    p, rest = pop_pair(unpaired, 1, i)
    return pair_partitions(push!(copy(pairs), p), rest)
  end |> Iterators.flatten |> collect
end

"""
$(TYPEDSIGNATURES)

Return all partitions of the vertices into pairs.
"""
function pair_partitions(vertices::Vector{Int})
  @assert length(vertices) % 2 == 0
  pairs = PairVector()
  return pair_partitions(pairs, vertices)
end

"""
$(TYPEDSIGNATURES)

Return all partitions of ``\\{1,...,2n\\}`` into ``n`` pairs.
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
  map(2:length(unpaired)) do i
    p, rest = pop_pair(unpaired, 1, i)
    # NOTE each step adds one swap => flips parity
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
$(SIGNATURES)

Return double factorial ``n!!``
"""
function double_factorial(n)
  n == 0 && return 1
  n == 1 && return 1
  return n * double_factorial(n - 2)
end

"""
Diagram with a topology and tuple of pseudo particle interaction pair indices
"""
struct Diagram
  "Topology"
  topology::Topology
  "Pair indices"
  pair_idxs::Tuple{Vararg{Int64}}
end

end # module diagrammatics
