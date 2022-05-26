module diagrammatics

using DocStringExtensions

"""
$(TYPEDFIELDS)
"""
struct Topology
  order::Int
  pairs::Vector{Pair{Int,Int}}
end

end # module diagrammatics
