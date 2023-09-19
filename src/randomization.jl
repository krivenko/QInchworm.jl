module randomization

using DocStringExtensions

using Random: AbstractRNG
using Statistics: mean, var, std
using LinearAlgebra: norm

using QInchworm.ScrambledSobol: ScrambledSobolSeq

#
# Randomized quasi Monte Carlo
#

"""
$(TYPEDEF)

Parameters of the randomized qMC integration.

$(TYPEDFIELDS)
"""
struct RandomizationParams
    "Random Number Generator used to scramble the Sobol sequences"
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
Estimate mean and standard deviation using randomized qMC.
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
