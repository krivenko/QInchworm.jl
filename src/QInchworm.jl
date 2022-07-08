"""
Fermionic quantum impurity problem solving
using Quasi Monte Carlo and the inch-worm algorithm.
"""
module QInchworm

include("ppgf.jl")
include("configuration.jl")
include("qmc_integrate.jl")
include("diagrammatics.jl")
include("topology_eval.jl")

end # module QInchworm
