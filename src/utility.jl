module utility

import Keldysh
import Interpolations
using Interpolations: BoundaryCondition,
                      GridType,
                      inner_system_diags,
                      Woodbury,
                      WoodburyMatrices,
                      lut!

import Sobol: AbstractSobolSeq, SobolSeq, next!, ndims

"""
    Inverse of get_point(c::AbstractContour, ref)

    TODO: Ask Joseph to promote it to Keldysh.jl?
"""
function get_ref(c::Keldysh.AbstractContour, t::Keldysh.BranchPoint)
    ref = 0
    for b in c.branches
        lb = length(b)
        if t.domain == b.domain
            return ref + (t.ref * lb)
        else
            ref += lb
        end
    end
    @assert false
end

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
    dl,d,du = inner_system_diags(T,n,degree)
    d[1] = d[end] = -oneunit(T)
    du[1] = dl[end] = zero(T)

    specs = WoodburyMatrices.sparse_factors(T, n,
                                  (1, 3, oneunit(T)),
                                  (n, n-2, oneunit(T))
                                 )

    b = zeros(TC, n)
    b[1] = 2/(n-3) * degree.bc.left_derivative
    b[end] = -2/(n-3) * degree.bc.right_derivative

    Woodbury(lut!(dl, d, du), specs...), b
end

"""
    Quadratic spline on an equidistant grid that allows for
    incremental construction.
"""
struct IncrementalSpline{T<:Number}
    knots::AbstractRange{T}
    data::Vector{Complex{T}}
    der_data::Vector{Complex{T}}

    function IncrementalSpline(knots::AbstractRange{T}, val1::Complex{T}, der1::Complex{T}) where {T<:Number}
        data = Complex{T}[val1]
        sizehint!(data, length(knots))
        der_data = Complex{T}[der1 * step(knots)]
        sizehint!(der_data, length(knots)-1)
        return new{T}(knots, data, der_data)
    end
end

function extend!(spline::IncrementalSpline{T}, val) where {T<:Number}
   push!(spline.data, val)
   push!(spline.der_data, 2*(spline.data[end] - spline.data[end-1]) - spline.der_data[end])
end

function (spline::IncrementalSpline{T})(z) where {T<:Number}
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
mutable struct BetterSobolSeq{N} <: AbstractSobolSeq{N}
    seq::SobolSeq{N}
    init_pt_returned::Bool

    BetterSobolSeq(N::Int) = new{N}(SobolSeq(N), false)
end

function next!(bseq::BetterSobolSeq)
    if bseq.init_pt_returned
        next!(bseq.seq)
    else
        bseq.init_pt_returned = true
        zeros(Float64, ndims(bseq.seq))
    end
end

end
