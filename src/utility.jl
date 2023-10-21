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
# Authors: Igor Krivenko, Hugo U. R. Strand

"""
An assorted collection of utility types and functions.

# Exports
$(EXPORTS)
"""
module utility

using DocStringExtensions

import Interpolations
using Octavian: matmul

using Sobol: AbstractSobolSeq, SobolSeq, ndims
import Sobol: next!

using Serialization

using Keldysh; kd = Keldysh

export ph_conj

"""
    $(TYPEDEF)

Interpolations.jl addon: Implementation of the Neumann boundary conditions for the cubic
spline.

# Fields
$(TYPEDFIELDS)
"""
struct NeumannBC{GT <: Union{Interpolations.GridType, Nothing},
                 T <: Number
                 } <: Interpolations.BoundaryCondition
    "Grid type"
    gt::GT
    "Function derivative at the left boundary"
    left_derivative::T
    "Function derivative at the right boundary"
    right_derivative::T
end

"""
    $(TYPEDSIGNATURES)

Compute the system used to prefilter cubic spline coefficients when using the Neumann
boundary conditions.

# Parameters
- `T`, `TC`: Element types.
- `n`:       The number of rows in the data input.
- `degree`:  Interpolation degree information.

# Returns
Woodbury matrix and the RHS of the computed system.
"""
function Interpolations.prefiltering_system(::Type{T},
                                            ::Type{TC},
                                            n::Int,
                                            degree::Interpolations.Cubic{BC}) where {
    T, TC, BC <: NeumannBC{Interpolations.OnGrid}}
    dl, d, du = Interpolations.inner_system_diags(T, n, degree)
    d[1] = d[end] = -oneunit(T)
    du[1] = dl[end] = zero(T)

    specs = Interpolations.WoodburyMatrices.sparse_factors(T, n,
                                                           (1, 3, oneunit(T)),
                                                           (n, n - 2, oneunit(T))
                                                           )

    b = zeros(TC, n)
    b[1] = 2 / (n - 3) * degree.bc.left_derivative
    b[end] = -2 / (n - 3) * degree.bc.right_derivative

    Interpolations.Woodbury(Interpolations.lut!(dl, d, du), specs...), b
end

"""
    $(TYPEDEF)

Quadratic spline on an equidistant grid that allows for incremental construction.

# Fields
$(TYPEDFIELDS)
"""
struct IncrementalSpline{KnotT <: Number, T <: Number}
    "Locations of interpolation knots"
    knots::AbstractRange{KnotT}
    "Values of the interpolated function at the knots"
    data::Vector{T}
    "Values of the interpolated function derivative at the knots"
    der_data::Vector{T}

    @doc """
        $(TYPEDSIGNATURES)

    Initialize an incremental spline and add the first segment to it by fixing values
    of the interpolated function `val1` and its derivative `der1` at the first knot.
    """
    function IncrementalSpline(knots::AbstractRange{KnotT}, val1::T, der1::T) where {
        KnotT <: Number, T <: Number}
        data = T[val1]
        sizehint!(data, length(knots))
        der_data = T[der1 * step(knots)]
        sizehint!(der_data, length(knots) - 1)
        return new{KnotT,T}(knots, data, der_data)
    end
end

"""
    $(TYPEDSIGNATURES)

Add a segment to an incremental spline by fixing value `val` of the interpolated function
at the next knot.
"""
function extend!(spline::IncrementalSpline, val)
   push!(spline.data, val)
   push!(spline.der_data,
         2 * (spline.data[end] - spline.data[end - 1]) - spline.der_data[end])
end

"""
    $(TYPEDSIGNATURES)

Evaluate spline interpolant at a point `z`.
"""
function (spline::IncrementalSpline)(z)
    @boundscheck first(spline.knots) <= z <= last(spline.knots)
    x = 1 + (z - first(spline.knots)) / step(spline.knots)
    i = floor(Int, x)
    i = min(i, length(spline.data) - 1)
    δx = x - i
    @inbounds c3 = spline.der_data[i] - 2 * spline.data[i + 1]
    @inbounds spline.data[i] * (1 - δx^2) +
              spline.data[i + 1] * (1 - (1 - δx)^2) +
              c3 * (0.25 - (δx - 0.5)^2)
end

"""
    $(TYPEDEF)

Sobol low discrepancy sequence including the initial point ``(0, 0, \\ldots)``.
The need for this modified version of `Sobol.SobolSeq` is discussed in
[JuliaMath/Sobol.jl#31](https://github.com/stevengj/Sobol.jl/issues/31).
"""
mutable struct SobolSeqWith0{N} <: AbstractSobolSeq{N}
    seq::SobolSeq{N}
    init_pt_returned::Bool

    SobolSeqWith0(N::Int) = new{N}(SobolSeq(N), false)
end

"""
    $(TYPEDSIGNATURES)

Return the next point in the Sobol sequence `s`.
"""
function next!(s::SobolSeqWith0)
    if s.init_pt_returned
        return next!(s.seq)
    else
        s.init_pt_returned = true
        return zeros(Float64, ndims(s.seq))
    end
end

"""
    $(TYPEDSIGNATURES)

Skip exactly `n` points in the Sobol sequence `s`.
"""
function arbitrary_skip!(s::SobolSeqWith0, n::Integer)
    @assert n >= 0
    if n >= 1
        s.init_pt_returned = true
        x = Array{Float64,1}(undef, ndims(s.seq))
        for _ = 1:(n - 1); next!(s.seq, x); end
    end
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Return a vector of `n` integers which are approximately equal and sum to `N`.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i = 1:n]
end

"""
    $(TYPEDSIGNATURES)

Given a list of chunk sizes, return the range that enumerates elements
in the `idx`-th chunk.
"""
function range_from_chunks_and_idx(chunk_sizes::AbstractVector, idx::Integer)
    sidx = 1 + sum(chunk_sizes[1:idx - 1])
    eidx = sidx + chunk_sizes[idx] - 1
    return sidx:eidx
end

"""
    $(SIGNATURES)

Serialize data using an `IOBuffer` object.
"""
function iobuffer_serialize(data)
    io = IOBuffer()
    serialize(io, data)
    seekstart(io)
    data_raw = read(io)
    close(io)
    return data_raw
end

"""
    $(SIGNATURES)

Deserialize data using an `IOBuffer` object.
"""
function iobuffer_deserialize(data_raw)
    return deserialize(IOBuffer(data_raw))
end

"""
    $(TYPEDSIGNATURES)

Given a scalar-valued imaginary time Green's function ``g(\\tau)``, return its particle-hole
conjugate ``g(\\beta-\\tau)``.
"""
function ph_conj(g::kd.ImaginaryTimeGF{T, true})::kd.ImaginaryTimeGF{T, true} where {T}
    g_rev = deepcopy(g)
    τ_0, τ_β = g.grid[1], g.grid[end]
    for τ in g.grid
        g_rev[τ, τ_0] = g[τ_β, τ]
    end
    return g_rev
end

#
# LazyMatrixProduct
#

"""
    $(TYPEDEF)

A matrix product of the form ``A_N A_{N-1} \\ldots A_1``.

Functions [`pushfirst!()`](@ref) and [`popfirst!()`](@ref) can be used to add and remove
multipliers to/from the left of the product. The product is lazy in the sense that the
actual multiplication takes place only when the [`eval!()`](@ref) function is called.
The structure keeps track of previously evaluated partial products and reuses them upon
successive calls to [`eval!()`](@ref).
"""
mutable struct LazyMatrixProduct{T <: Number}
    "Multipliers ``A_i``"
    matrices::Vector{Matrix{T}}
    "Partial products ``A_1, A_2 A_1, A_3 A_2 A_1, \\ldots``"
    partial_prods::Vector{Matrix{T}}
    "Current number of matrices in the product"
    n_mats::Int
    "Current number of evaluated partial products that are still valid"
    n_prods::Int
end

"""
    $(TYPEDSIGNATURES)

Make a [`LazyMatrixProduct`](@ref) instance and pre-allocate all necessary containers.

# Parameters
- `T`:           Element type of the matrices to be multiplied.
- `max_n_mats` : Maximal number of matrices in the product used to pre-allocate containers.
"""
function LazyMatrixProduct(::Type{T}, max_n_mats::Int) where {T <: Number}
    @assert max_n_mats > 0
    LazyMatrixProduct(
        Vector{Matrix{T}}(undef, max_n_mats),
        Vector{Matrix{T}}(undef, max_n_mats),
        0, 0
    )
end

"""
    $(TYPEDSIGNATURES)

Add a new matrix `A` to the left of the product `lmp`.
"""
function Base.pushfirst!(lmp::LazyMatrixProduct{T}, A::Matrix{T}) where {T <: Number}
    @boundscheck lmp.n_mats < length(lmp.matrices)
    lmp.n_mats += 1
    lmp.matrices[lmp.n_mats] = A
end

"""
    $(TYPEDSIGNATURES)

Remove `n` matrices from the left of the product `lmp`. By default, `n = 1`.
"""
function Base.popfirst!(lmp::LazyMatrixProduct{T}, n::Int = 1) where {T <: Number}
    @boundscheck n <= lmp.n_mats
    lmp.n_mats -= n
    lmp.n_prods = min(lmp.n_prods, lmp.n_mats)
end

"""
    $(TYPEDSIGNATURES)

Evaluate the matrix product `lmp`.
"""
function eval!(lmp::LazyMatrixProduct{T}) where {T <: Number}
    @boundscheck lmp.n_mats > 0

    # The first partial product is simply A_1
    if lmp.n_prods == 0
        lmp.partial_prods[1] = lmp.matrices[1]
        lmp.n_prods = 1
    end

    # Do the multiplication
    for n = (lmp.n_prods + 1):lmp.n_mats
        lmp.partial_prods[n] = matmul(lmp.matrices[n], lmp.partial_prods[n - 1])
    end

    lmp.n_prods = lmp.n_mats

    return lmp.partial_prods[lmp.n_prods]
end

end # module utility
