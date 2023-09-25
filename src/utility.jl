module utility

import Interpolations
using Octavian

using Sobol: AbstractSobolSeq, SobolSeq, ndims
import Sobol: next!

using Serialization

#
# Interpolations.jl addon: Implementation of the Neumann boundary
# conditions for the cubic spline.
#

struct NeumannBC{GT<:Union{Interpolations.GridType, Nothing}, T<:Number} <: Interpolations.BoundaryCondition
    gt::GT
    left_derivative::T
    right_derivative::T
end

function Interpolations.prefiltering_system(::Type{T},
                                            ::Type{TC},
                                            n::Int,
                                            degree::Interpolations.Cubic{BC}) where {
    T, TC, BC<:NeumannBC{Interpolations.OnGrid}}
    dl,d,du = Interpolations.inner_system_diags(T,n,degree)
    d[1] = d[end] = -oneunit(T)
    du[1] = dl[end] = zero(T)

    specs = Interpolations.WoodburyMatrices.sparse_factors(T, n,
                                                           (1, 3, oneunit(T)),
                                                           (n, n-2, oneunit(T))
                                                           )

    b = zeros(TC, n)
    b[1] = 2/(n-3) * degree.bc.left_derivative
    b[end] = -2/(n-3) * degree.bc.right_derivative

    Interpolations.Woodbury(Interpolations.lut!(dl, d, du), specs...), b
end

"""
    Quadratic spline on an equidistant grid that allows for
    incremental construction.
"""
struct IncrementalSpline{KnotT<:Number, T<:Number}
    knots::AbstractRange{KnotT}
    data::Vector{T}
    der_data::Vector{T}

    function IncrementalSpline(knots::AbstractRange{KnotT}, val1::T, der1::T) where {KnotT<:Number, T<:Number}
        data = T[val1]
        sizehint!(data, length(knots))
        der_data = T[der1 * step(knots)]
        sizehint!(der_data, length(knots)-1)
        return new{KnotT,T}(knots, data, der_data)
    end
end

function extend!(spline::IncrementalSpline, val)
   push!(spline.data, val)
   push!(spline.der_data, 2*(spline.data[end] - spline.data[end-1]) - spline.der_data[end])
end

function (spline::IncrementalSpline)(z)
    @assert first(spline.knots) <= z <= last(spline.knots)
    x = 1 + (z - first(spline.knots)) / step(spline.knots)
    i = floor(Int, x)
    i = min(i, length(spline.data) - 1)
    δx = x - i
    @inbounds c3 = spline.der_data[i] - 2 * spline.data[i + 1]
    @inbounds spline.data[i] * (1-δx^2) + spline.data[i + 1] * (1-(1-δx)^2) + c3 * (0.25-(δx-0.5)^2)
end

"""
    Sobol sequence including the initial point (0, 0, ...)

    C.f. https://github.com/stevengj/Sobol.jl/issues/31
"""
mutable struct SobolSeqWith0{N} <: AbstractSobolSeq{N}
    seq::SobolSeq{N}
    init_pt_returned::Bool

    SobolSeqWith0(N::Int) = new{N}(SobolSeq(N), false)
end

function next!(s::SobolSeqWith0)
    if s.init_pt_returned
        next!(s.seq)
    else
        s.init_pt_returned = true
        zeros(Float64, ndims(s.seq))
    end
end

function arbitrary_skip!(s::SobolSeq, n::Integer)
    x = Array{Float64,1}(undef, ndims(s))
    for unused = 1:n
        next!(s,x)
    end
    return nothing
end

function arbitrary_skip!(s::SobolSeqWith0, n::Integer)
    @assert n >= 0
    if n >= 1
        s.init_pt_returned = true
        arbitrary_skip!(s.seq, n - 1)
    end
    return nothing
end

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
"""
function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

function range_from_chunks_and_idx(chunks, idx)
    sidx = 1 + sum(chunks[1:idx-1])
    eidx = sidx + chunks[idx] - 1
    return sidx:eidx
end

function iobuffer_serialize(data)
    io = IOBuffer()
    serialize(io, data)
    seekstart(io)
    data_raw = read(io)
    close(io)
    return data_raw
end

function iobuffer_deserialize(data_raw)
    return deserialize(IOBuffer(data_raw))
end

#
# LazyMatrixProduct
#

"""
A matrix product of the form A_N A_{N-1} ... A_1.

Functions `pushfirst!()` and `popfirst!()` can be used to add and remove multipliers
to/from the left of the product. The product is lazy in the sense that the actual
multiplication takes place only when the `eval!()` function is called. The structure keeps
track of previously evaluated partial products and reuses them upon successive calls to
`eval!()`.
"""
mutable struct LazyMatrixProduct{T <: Number}
    "Multipliers A_i"
    matrices::Vector{AbstractMatrix{T}}
    "Pre-allocated 1D buffers for partial products A_1, A_2 A_1, A_3 A_2 A_1, ..."
    partial_prods::Array{T, 2}
    "Current number of matrices in the product"
    n_mats::Int
    "Current number of evaluated partial products that are still valid"
    n_prods::Int
end

"""
Make a `LazyMatrixProduct` instance and pre-allocate all necessary buffers.

T          : Element type of the matrices to be multiplied.
max_n_mats : Maximal number of matrices in the product used to pre-allocate buffers.
max_dim    : Maximal dimension of a matrix in the product used to pre-allocate buffers.
"""
function LazyMatrixProduct(::Type{T}, max_n_mats::Int, max_dim::Int) where {T <: Number}
    @assert max_n_mats > 0
    @assert max_dim > 0
    LazyMatrixProduct(
        Vector{AbstractMatrix{T}}(undef, max_n_mats),
        Array{T, 2}(undef, max_dim^2, max_n_mats),
        0, 0
    )
end

"""
Add a new matrix to the left of the product.
"""
function Base.pushfirst!(lmp::LazyMatrixProduct{T}, A::AbstractMatrix{T}) where {T <: Number}
    @boundscheck lmp.n_mats < length(lmp.matrices)
    lmp.n_mats += 1
    lmp.matrices[lmp.n_mats] = A
end

"""
Remove `n` matrices from the left of the product.
"""
function Base.popfirst!(lmp::LazyMatrixProduct{T}, n::Int = 1) where {T <: Number}
    @boundscheck n <= lmp.n_mats
    lmp.n_mats -= n
    lmp.n_prods = min(lmp.n_prods, lmp.n_mats)
end

"""
Evaluate the product.
"""
function eval!(lmp::LazyMatrixProduct{T}) where {T <: Number}
    @boundscheck lmp.n_mats > 0

    # Second dimension of the rightmost matrix = second dimension of the whole product
    r_dim::Int = size(lmp.matrices[1], 2)

    # The first partial product is simply A_1
    if lmp.n_prods == 0
        lmp.partial_prods[1:length(lmp.matrices[1]), 1] = lmp.matrices[1][:]
        lmp.n_prods = 1
    end

    # Do the multiplication
    for n = (lmp.n_prods + 1):lmp.n_mats
        l_dim::Int = size(lmp.matrices[n - 1], 1)
        l_dim_new::Int = size(lmp.matrices[n], 1)

        Octavian.matmul!(
            reshape(view(lmp.partial_prods, 1:l_dim_new * r_dim, n), l_dim_new, r_dim),
            lmp.matrices[n],
            reshape(view(lmp.partial_prods, 1:l_dim * r_dim, n - 1), l_dim, r_dim)
        )
    end

    lmp.n_prods = lmp.n_mats

    l_dim = size(lmp.matrices[lmp.n_mats], 1)
    return reshape(view(lmp.partial_prods, 1:l_dim * r_dim, lmp.n_prods), l_dim, r_dim)
end

end # module utility
