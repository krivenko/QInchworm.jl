# This module contains a modified and extended version of Sobol.jl's code written by
# Steven G. Johnson (https://github.com/JuliaMath/Sobol.jl/)
#
# The extensions include those mentioned in Sobol.jl issue #31
# https://github.com/stevengj/Sobol.jl/issues/31
#
# - Support for scrambled Sobol points (implementation of the scrambling is taken from
#   SciPy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).
# - Inclusion of the initial point in the Sobol sequence.
# - skip!() now skips 2^m points.

module ScrambledSobol

using Random
using LinearAlgebra: dot, LowerTriangular

export ScrambledSobolSeq, next!

include("soboldata.jl") # Loads `sobol_a` and `sobol_minit`

# D is the dimension of sequence being generated
mutable struct ScrambledSobolSeq{D}
    m::Array{UInt32,2}     # Direction numbers, array of size (D, 32)
    x::Array{UInt32,1}     # Previous x = x_n, array of length D
    n::UInt32              # Number of x's generated so far
end

function ScrambledSobolSeq(D::Int; scramble_rng::Union{AbstractRNG, Nothing} = nothing)
    (D < 0 || D > (length(sobol_a) + 1)) && error("Invalid Sobol dimension $(D)")

    m = ones(UInt32, (D, 32))

    # Special case D = 0
    D == 0 && return(ScrambledSobolSeq{0}(m, UInt32[], zero(UInt32)))

    #
    # Initialize matrix of the direction numbers
    #

    for d in 2:D
        a = sobol_a[d - 1]
        deg = floor(Int, log2(a)) # Degree of polynomial

        # Set initial values of m from table
        m[d, 1:deg] = sobol_minit[1:deg, d - 1]
        # Fill in remaining values using recurrence
        for j = (deg + 1):32
            ac = a
            m[d, j] = m[d, j-deg]
            for k = 0:deg-1
                @inbounds m[d, j] = m[d, j] ⊻ (((ac & one(UInt32)) * m[d, j-deg+k]) << (deg-k))
                ac >>= 1
            end
        end
    end

    # Multiply each column of m by power of 2:
    # m * [2^31, 2^30, ... , 2, 1]
    for j in 1:32
        m[:, j] *= UInt32(2) .^ (32 - j)
    end

    if scramble_rng === nothing
        x = zeros(UInt32, D)
    else # Scramble the sequence using LMS+shift
        # Generate shift vector
        x = rand(scramble_rng, UInt32[0, 1], D, 32) * (UInt32(2) .^ (0:31))

        # Generate lower triangular matrices (stacked across dimensions)
        ltm = mapslices(LowerTriangular,
                        rand(scramble_rng, UInt32[0, 1], D, 32, 32);
                        dims=(2, 3))

        # Set diagonals of ltm to 1
        for d in 1:D
            for i in 1:32
                ltm[d, i, i] = one(eltype(ltm))
            end
        end

        # Apply the linear matrix scramble
        for d in 1:D
            for j in 1:32
                mdj = m[d, j]
                l = 1
                t2 = 0
                for p in 32:-1:1
                    lsmdp = dot(ltm[d, p, :], UInt32(2) .^ (31:-1:0))
                    t1 = 0
                    for k in 1:32
                        t1 += ((lsmdp >> (k - 1)) & one(UInt32)) *
                              ((mdj >> (k - 1)) & one(UInt32))
                    end
                    t1 = t1 % 2
                    t2 = t2 + t1 * l
                    l = 2 * l
                end
                m[d, j] = t2
            end
        end
    end

    return ScrambledSobolSeq{D}(m, x, zero(UInt32))
end

Base.ndims(s::ScrambledSobolSeq{D}) where {D} = D::Int

function next!(s::ScrambledSobolSeq, x::AbstractVector{<:AbstractFloat})
    length(x) != ndims(s) && throw(BoundsError())

    if s.n == zero(s.n)
        s.n += one(s.n)
        x[:] = ldexp.(Float64.(s.x), -32)
        return x
    end

    s.n += one(s.n)
    c = UInt32(trailing_zeros(s.n - 1))
    s.x[:] = s.x .⊻ s.m[:, c + 1]
    x[:] = ldexp.(Float64.(s.x), -32)

    return x
end
next!(s::ScrambledSobolSeq) = next!(s, Array{Float64,1}(undef, ndims(s)))

# skip!(s, n) skips 2^m such that 2^m < n ≤ 2^(m+1)
# skip!(s, n, exact=true) skips m = n
function skip!(s::ScrambledSobolSeq, n::Integer, x; exact=false)
    if n ≤ 0
        n == 0 && return s
        throw(ArgumentError("$n is not non-negative"))
    end
    nskip = exact ? n : (1 << floor(Int, log2(n + 1)))
    for _ = 1:nskip; next!(s, x); end
    return s
end
function skip!(s::ScrambledSobolSeq, n::Integer; exact=false)
    return skip!(s, n, Array{Float64,1}(undef, ndims(s)); exact=exact)
end

function Base.show(io::IO, s::ScrambledSobolSeq)
    print(io, "$(ndims(s))-dimensional scrambled Sobol sequence on [0,1]^$(ndims(s))")
end
function Base.show(io::IO, ::MIME"text/html", s::ScrambledSobolSeq)
    print(io, "$(ndims(s))-dimensional scrambled Sobol sequence on [0,1]<sup>$(ndims(s))</sup>")
end

end # module ScrambledSobol
