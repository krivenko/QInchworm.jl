# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Authors: Igor Krivenko, Hugo U. R. Strand, Joseph Kleinhenz

"""
A quasi Monte Carlo inchworm impurity solver for multi-orbital fermionic models.
"""
module QInchworm

include("scrambled_sobol.jl")
include("utility.jl")
include("sector_block_matrix.jl")
include("mpi.jl")
include("diagrammatics.jl")
include("spline_gf.jl")
include("ppgf.jl")
include("expansion.jl")
include("configuration.jl")
include("qmc_integrate.jl")
include("randomization.jl")
include("topology_eval.jl")
include("inchworm.jl")

end # module QInchworm
