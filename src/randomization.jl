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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/.
#
# Author: Igor Krivenko

"""
Randomized quasi Monte Carlo.

# Exports
$(EXPORTS)
"""
module randomization

using DocStringExtensions

using Random: AbstractRNG
using Statistics: mean, var, std
using LinearAlgebra: norm

using QInchworm.ScrambledSobol: ScrambledSobolSeq

export RandomizationParams

"""
    $(TYPEDEF)

Parameters of the randomized qMC integration.

# Fields
$(TYPEDFIELDS)
"""
struct RandomizationParams
    """
    Random Number Generator used to scramble Sobol sequences
    or `nothing` to disable scrambling
    """
    rng::Union{AbstractRNG, Nothing}
    "Maximal number of scrambled Sobol sequences to be used"
    N_seqs::Int64
    "Target standard deviation of the accumulated quantity (∞-norm for matrices)"
    target_std::Float64
end
RandomizationParams() = RandomizationParams(nothing, 1, .0)

function Base.show(io::IO, rp::RandomizationParams)
    print(io,
    "RandomizationParams(rng=$(rp.rng), N_seqs=$(rp.N_seqs), target_std=$(rp.target_std))")
end

"""
    $(TYPEDSIGNATURES)

Estimate mean and standard deviation using randomized quasi Monte Carlo.

This function initializes a number of `D`-dimensional
[scrambled Sobol sequences](@ref ScrambledSobolSeq) using the
[set of parameters](@ref RandomizationParams) `params` and passes each of them to
the function `f`. It returns the mean and the standard deviation of `f`'s return values.
"""
function mean_std_from_randomization(f::Function, D::Int, params::RandomizationParams)
    @assert params.N_seqs > 0
    samples = Vector{first(Base.return_types(f, (ScrambledSobolSeq,)))}()
    for s = 1:params.N_seqs
        let seq = ScrambledSobolSeq(D, scramble_rng = params.rng)
            push!(samples, f(seq))
            s > 1 && norm(std(samples), Inf) <= params.target_std && break
        end
    end
    return (mean(samples), std(samples))
end

end # module randomization
