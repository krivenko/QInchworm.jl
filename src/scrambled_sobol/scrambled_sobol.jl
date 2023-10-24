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
# Authors: Steven G. Johnson, Igor Krivenko

"""
This module contains a modified and extended version of
[`Sobol.jl`'s](https://github.com/JuliaMath/Sobol.jl/) code written by Steven G. Johnson.

The extensions include those mentioned in
[Sobol.jl issue #31](https://github.com/stevengj/Sobol.jl/issues/31).

- Support for scrambled Sobol points (implementation of the scrambling is taken from
  [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)).
- Inclusion of the initial point in the Sobol sequence.
- `skip!(... ; exact=false)` now skips ``2^m`` points.
"""
module scrambled_sobol

using DocStringExtensions

using Random
using LinearAlgebra: dot, LowerTriangular

export ScrambledSobolSeq, next!

include("soboldata.jl") # Loads `sobol_a` and `sobol_minit`

"""
    $(TYPEDEF)

Scrambled Sobol low-discrepancy sequence of dimension `D`.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ScrambledSobolSeq{D}
    "Direction numbers, array of size (`D`, 32)"
    m::Array{UInt32,2}
    "Previous sequence point ``x = x_n``, array of length `D`"
    x::Array{UInt32,1}
    "Number of sequence points generated so far"
    n::UInt32
end

"""
    $(TYPEDSIGNATURES)

Create a scrambled Sobol sequence of dimension `D` using the Random Number Generator
`scramble_rng` to generate scrambling parameters. The sequence remains unscrambled if no RNG
is provided.
"""
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
                @inbounds m[d, j] = m[d, j] ⊻
                                    (((ac & one(UInt32)) * m[d, j-deg+k]) << (deg-k))
                ac >>= 1
            end
        end
    end

    # Multiply each column of m by power of 2:
    # m * [2^31, 2^30, ... , 2, 1]
    for j in 1:32
        m[:, j] *= UInt32(2) .^ (32 - j)
    end

    if isnothing(scramble_rng)
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

"""
    $(TYPEDSIGNATURES)

Dimension `D` of a scrambled Sobol sequence.
"""
Base.ndims(s::ScrambledSobolSeq{D}) where {D} = D::Int

"""
    $(TYPEDSIGNATURES)

Generate the next point ``\\mathbf{x}\\in[0, 1]^D`` of the sequence `s` and write it into
the array `x`.
"""
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
"""
    $(TYPEDSIGNATURES)

Generate and return the next point ``\\mathbf{x}\\in[0, 1]^D`` of the sequence `s`.
"""
next!(s::ScrambledSobolSeq) = next!(s, Array{Float64,1}(undef, ndims(s)))

"""
    $(TYPEDSIGNATURES)

Skip a number of points in the sequence `s` using a preallocated buffer `x`.

- `skip!(s, n, x)` skips the next ``2^m`` points such that ``2^m < n \\leq 2^{m+1}``.
- `skip!(s, n, x, exact=true)` skips the next ``n`` points.
"""
function skip!(s::ScrambledSobolSeq, n::Integer, x; exact=false)
    if n ≤ 0
        n == 0 && return s
        throw(ArgumentError("$n is not non-negative"))
    end
    nskip = exact ? n : (1 << floor(Int, log2(n + 1)))
    for _ = 1:nskip; next!(s, x); end
    return s
end
"""
    $(TYPEDSIGNATURES)

Skip a number of points in the sequence `s`.

- `skip!(s, n)` skips the next ``2^m`` points such that ``2^m < n \\leq 2^{m+1}``.
- `skip!(s, n, exact=true)` skips the next ``n`` points.
"""
function skip!(s::ScrambledSobolSeq, n::Integer; exact=false)
    return skip!(s, n, Array{Float64,1}(undef, ndims(s)); exact=exact)
end

function Base.show(io::IO, s::ScrambledSobolSeq)
    print(io, "$(ndims(s))-dimensional scrambled Sobol sequence on [0,1]^$(ndims(s))")
end
function Base.show(io::IO, ::MIME"text/html", s::ScrambledSobolSeq)
    print(io,
          "$(ndims(s))-dimensional scrambled Sobol sequence on [0,1]<sup>$(ndims(s))</sup>")
end

end # module scrambled_sobol
