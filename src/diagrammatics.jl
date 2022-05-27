module diagrammatics

using DocStringExtensions

"""
$(TYPEDEF)

Datatype for diagram topology.
A topology of order ``n`` consists of a partition of ``\\{1,...,2n\\}`` into ``n`` pairs.
Diagrammatically a topology can be thought of as a set of arcs connecting vertices located at ``\\{1,...,2n\\}``.

$(TYPEDFIELDS)
"""
struct Topology
  order::Int
  pairs::Vector{Pair{Int,Int}}

  function Topology(pairs::Vector{Pair{Int,Int}})
    new(length(pairs), pairs)
  end
end

function Base.isvalid(t::Topology)
  sort!(collect(Iterators.Flatten(t.pairs))) == 1:2*t.order
end

function sortpair(p::Pair{T,T}) where T
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

function _isconnected!(connected::Vector{Pair{Int,Int}}, disconnected::Vector{Pair{Int,Int}})
  while !isempty(connected)
    p_c = pop!(connected)
    # pairs which cross with p_c are connected
    for (i, p_dc) in Iterators.reverse(enumerate(disconnected))
      iscrossing(p_c, p_dc) && push!(connected, popat!(disconnected, i))
    end
  end
  return isempty(disconnected)
end

function _isconnected(pairs::Vector{Pair{Int,Int}}, k::Int)
  connected = Pair{Int,Int}[]
  disconnected = Pair{Int,Int}[]

  for p in pairs
    if p.first <= k || p.second <= k
      push!(connected, p)
    else
      push!(disconnected, p)
    end
  end

  return _isconnected!(connected, disconnected)
end

"""
$(TYPEDSIGNATURES)

Given a topology, check if every connected component of the graph induced by
crossings between the arcs contains a pair with index <= `k`
"""
isconnected(t::Topology, k::Int) = _isconnected(t.pairs, k)

"""
Datatype for a labeled topology.
A labeled topology is a topology together with an operator for each vertex
"""
struct LabeledTopology
#TODO
end

# TODO given a vector of iteraction pairs, generated all possible labeled topologies

end # module diagrammatics
