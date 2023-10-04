"""
A quasi Monte Carlo inchworm impurity solver for multi-orbital fermionic models.
"""
module QInchworm

include("utility.jl")
include("sector_block_matrix.jl")
include("mpi.jl")
include("diagrammatics.jl")
include("spline_gf.jl")
include("ppgf.jl")
include("expansion.jl")
include("configuration.jl")
include("qmc_integrate.jl")
include("topology_eval.jl")
include("inchworm.jl")

end # module QInchworm
