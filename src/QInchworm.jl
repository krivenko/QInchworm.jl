"""
Fermionic quantum impurity problem solving
using Quasi Monte Carlo and the inch-worm algorithm.
"""
module QInchworm

include("utility.jl")
include("diagrammatics.jl")
include("spline_gf.jl")
include("ppgf.jl")
include("sector_block_matrix.jl")
include("expansion.jl")
include("configuration.jl")
include("qmc_integrate.jl")
include("topology_eval.jl")
include("inchworm.jl")

include("KeldyshED_addons.jl")

end # module QInchworm
