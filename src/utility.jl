module utility

using MPI: MPI

using Serialization

import Interpolations

using Sobol: AbstractSobolSeq, SobolSeq, ndims
import Sobol: next!

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

function arbitrary_skip(s::SobolSeq, n::Integer)
    x = Array{Float64,1}(undef, ndims(s))
    for unused = 1:n
        next!(s,x)
    end
    return nothing
end

function arbitrary_skip(s::SobolSeqWith0, n::Integer)
    @assert n >= 0
    if n >= 1
        s.init_pt_returned = true
        arbitrary_skip(s.seq, n-1)
    end
end

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
"""
function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

function mpi_N_skip_and_N_samples_on_rank(N_samples)

    comm_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    comm_size = MPI.Comm_size(MPI.COMM_WORLD)
    N_split = split_count(N_samples, comm_size)
    N_skip = sum(N_split[1:comm_rank])
    N_samples_on_rank = N_split[comm_rank+1]
    return N_skip, N_samples_on_rank

end

function inch_print()
    return MPI.Comm_rank(MPI.COMM_WORLD) == 0
end


function range_from_chuncks_and_idx(chunks, idx)
    sidx = 1 + sum(chunks[1:idx-1])
    eidx = sidx + chunks[idx] - 1
    return sidx:eidx
end

function rank_sub_range(n)
    comm_size = MPI.Comm_size(MPI.COMM_WORLD)
    comm_rank = MPI.Comm_rank(MPI.COMM_WORLD) # zero based indexing
    chunks = split_count(n, comm_size)
    return range_from_chuncks_and_idx(chunks, comm_rank + 1)
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

function mpi_all_gather_julia_vector(subvec::Vector{T}; comm = MPI.COMM_WORLD) where T
    data_raw = iobuffer_serialize(subvec)
    data_size = length(data_raw)

    size = [data_size]
    sizes = MPI.Allgather(size, comm)

    output = zeros(UInt8, sum(sizes))
    output_vbuf = MPI.VBuffer(output, sizes)

    MPI.Allgatherv!(data_raw, output_vbuf, comm)

    out_vec = T[]
    for i = 1:length(sizes)
        r = range_from_chuncks_and_idx(sizes, i)
        subvec_i = iobuffer_deserialize(output[r])
        append!(out_vec, subvec_i)
    end
    return out_vec
end

end
