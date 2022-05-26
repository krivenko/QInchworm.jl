"""
Fermionic quantum impurity problem solving
using Quasi Monte Carlo and the inch-worm algorithm.
"""
module QInchworm

using Requires

include("ppgf.jl")
include("configuration.jl")
include("qmc_integrate.jl")
include("diagrammatics.jl")

function __init__()
  # runs if PyPlot is loaded
  @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" include("pyplot.jl")
end

end # module QInchworm
