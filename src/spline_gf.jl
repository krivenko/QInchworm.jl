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
# Authors: Igor Krivenko, Hugo U. R. Strand

module spline_gf

using Interpolations: BSplineInterpolation,
                      scale,
                      BSpline,
                      Cubic,
                      Line,
                      OnGrid
import Interpolations: interpolate

using Keldysh; kd = Keldysh;

using QInchworm.utility: IncrementalSpline
import QInchworm.utility: extend!
import QInchworm.utility: ph_conj

#
# SplineInterpolatedGF
#

"""
Wrapper around a Green's function object that allows for
fast cubic spline interpolation on the time grid.
"""
struct SplineInterpolatedGF{GFType, T, scalar} <: kd.AbstractTimeGF{T, scalar}
    "Wrapped Green's function"
    GF::GFType
    "B-spline interpolants, one object per matrix element of G"
    interpolants
end

function SplineInterpolatedGF(
    GF::GFType;
    τ_max::kd.TimeGridPoint = GF.grid[end]) where {
        T <: Number, scalar, GFType <: kd.AbstractTimeGF{T, scalar}
    }
    norb = kd.norbitals(GF)
    interpolants = [make_interpolant(GF, k, l, τ_max) for k=1:norb, l=1:norb]
    return SplineInterpolatedGF{GFType, T, scalar}(GF, interpolants)
end

Base.eltype(::Type{<:SplineInterpolatedGF{GFType}}) where GFType = eltype(GFType)
Base.eltype(X::SplineInterpolatedGF) = eltype(typeof(X))

function Base.getproperty(G_int::SplineInterpolatedGF, p::Symbol)
    return (p == :grid) ? G_int.GF.grid : getfield(G_int, p)
end

kd.norbitals(G_int::SplineInterpolatedGF) = kd.norbitals(G_int.GF)
kd.TimeDomain(G_int::SplineInterpolatedGF) = kd.TimeDomain(G_int.GF)

@inline function Base.getindex(G_int::SplineInterpolatedGF,
                               k, l,
                               t1::kd.TimeGridPoint, t2::kd.TimeGridPoint,
                               greater=true)
    return G_int.GF[k, l, t1, t2, greater]
end

@inline function Base.getindex(G_int::SplineInterpolatedGF,
                               t1::kd.TimeGridPoint, t2::kd.TimeGridPoint,
                               greater=true)
    G_int.GF[t1, t2, greater]
end

@inline function Base.setindex!(G_int::SplineInterpolatedGF,
                                v,
                                k, l,
                                t1::kd.TimeGridPoint, t2::kd.TimeGridPoint)
    G_int.GF[k, l, t1, t2] = v
    update_interpolant!(G_int, k, l)
    return v
end

@inline function Base.setindex!(G_int::SplineInterpolatedGF,
                                v,
                                t1::kd.TimeGridPoint, t2::kd.TimeGridPoint)
    G_int.GF[t1, t2] = v
    update_interpolants!(G_int)
    return v
end

function (G_int::SplineInterpolatedGF)(t1::kd.BranchPoint, t2::kd.BranchPoint)
    return interpolate(G_int, t1, t2)
end

#
# Imaginary time GF
#

function update_interpolant!(
    G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar},
    k, l;
    τ_max::kd.TimeGridPoint = G_int.GF.grid[end]) where {T <: Number, scalar}
    G_int.interpolants[k, l] = make_interpolant(G_int.GF, k, l, τ_max)
end

function update_interpolants!(
    G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar};
    τ_max::kd.TimeGridPoint = G_int.GF.grid[end]) where {T <: Number, scalar}
    norb = kd.norbitals(G_int)
    for k=1:norb, l=1:norb
        update_interpolant!(G_int, k, l, τ_max=τ_max)
    end
end

@inline function Base.setindex!(
    G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar},
    v,
    k, l,
    t1::kd.TimeGridPoint, t2::kd.TimeGridPoint;
    τ_max::kd.TimeGridPoint = G_int.GF.grid[end]) where {T <: Number, scalar}
    G_int.GF[k, l, t1, t2] = v
    update_interpolant!(G_int, k, l, τ_max=τ_max)
    return v
end

@inline function Base.setindex!(
    G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar},
    v,
    t1::kd.TimeGridPoint, t2::kd.TimeGridPoint;
    τ_max::kd.TimeGridPoint = G_int.GF.grid[end]) where {T <: Number, scalar}
    G_int.GF[t1, t2] = v
    update_interpolants!(G_int, τ_max=τ_max)
    return v
end

function make_interpolant(GF::kd.ImaginaryTimeGF{T, scalar},
                          k, l,
                          τ_max::kd.TimeGridPoint) where {T <: Number, scalar}
    grid = GF.grid
    Δτ = -imag(step(grid, kd.imaginary_branch))
    knots = LinRange(0, kd.get_ref(grid.contour, τ_max.bpoint), τ_max.cidx)
    return scale(
        interpolate(GF.mat.data[k, l, 1:length(knots)], BSpline(Cubic(Line(OnGrid())))),
        knots)
end

function interpolate(G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, true}, T, true},
                     t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = kd.get_ref(grid.contour, t1)
    ref2 = kd.get_ref(grid.contour, t2)

    return ref1 >= ref2 ? G_int.interpolants[1, 1](ref1 - ref2) :
                          Int(G_int.GF.ξ) * G_int.interpolants[1, 1](β + ref1 - ref2)
end

function interpolate(G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, false}, T, false},
    t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = kd.get_ref(grid.contour, t1)
    ref2 = kd.get_ref(grid.contour, t2)

    norb = kd.norbitals(G_int.GF)
    if ref1 >= ref2
        return [G_int.interpolants[k, l](ref1 - ref2) for k=1:norb, l=1:norb]
    else
        return Int(G_int.GF.ξ) * [G_int.interpolants[k, l](β + ref1 - ref2)
                                  for k=1:norb, l=1:norb]
    end
end

ph_conj(G_int::SplineInterpolatedGF) = SplineInterpolatedGF(ph_conj(G_int.GF))

#
# IncSplineImaginaryTimeGF
#

"""
Wrapper around an imaginary time Green's function object that
supports interpolation based on the IncrementalSpline.
"""
struct IncSplineImaginaryTimeGF{T, scalar} <: kd.AbstractTimeGF{T, scalar}
    "Wrapped Green's function"
    GF::kd.ImaginaryTimeGF{T, scalar}
    "Incremental spline interpolants, one object per matrix element of G"
    interpolants

    function IncSplineImaginaryTimeGF(GF::kd.ImaginaryTimeGF{T, true},
                                      derivative_at_0::T) where {T <: Number}
        interpolants = [make_inc_interpolant(GF, k, l, derivative_at_0)]
        return new{T, true}(GF, interpolants)
    end

    function IncSplineImaginaryTimeGF(GF::kd.ImaginaryTimeGF{T, false},
                                      derivative_at_0::Matrix{T}) where {T <: Number}
        norb = kd.norbitals(GF)
        interpolants = [make_inc_interpolant(GF, k, l, derivative_at_0[k, l])
                        for k=1:norb, l=1:norb]
        return new{T, false}(GF, interpolants)
    end
end

function make_inc_interpolant(GF::kd.ImaginaryTimeGF{T, scalar},
                              k, l,
                              derivative_at_0::T) where {T <: Number, scalar}
    grid = GF.grid
    Δτ = -imag(step(grid, kd.imaginary_branch))
    knots = LinRange(0, kd.get_ref(grid.contour, grid[end].bpoint), grid[end].cidx)
    return IncrementalSpline(knots, GF.mat.data[k, l, 1], derivative_at_0)
end

Base.eltype(::Type{<:IncSplineImaginaryTimeGF{T, scalar}}) where {T, scalar} = T
Base.eltype(X::IncSplineImaginaryTimeGF) = eltype(typeof(X))

function Base.zero(G_int::IncSplineImaginaryTimeGF{T, false}) where {T <: Number}
    norb = kd.norbitals(G_int.GF)
    return IncSplineImaginaryTimeGF(kd.zero(G_int.GF), zeros(T, (norb, norb)))
end

function Base.zero(G_int::IncSplineImaginaryTimeGF{T, true}) where {T <: Number}
    return IncSplineImaginaryTimeGF(kd.zero(G_int.GF), zero(T))
end

function Base.getproperty(G_int::IncSplineImaginaryTimeGF, p::Symbol)
    return (p == :grid) ? G_int.GF.grid : getfield(G_int, p)
end

kd.norbitals(G_int::IncSplineImaginaryTimeGF) = kd.norbitals(G_int.GF)
kd.TimeDomain(G_int::IncSplineImaginaryTimeGF) = kd.TimeDomain(G_int.GF)

@inline function Base.getindex(G_int::IncSplineImaginaryTimeGF,
    k, l,
    t1::kd.TimeGridPoint, t2::kd.TimeGridPoint,
    greater=true)
    return G_int.GF[k, l, t1, t2, greater]
end

@inline function Base.getindex(G_int::IncSplineImaginaryTimeGF,
    t1::kd.TimeGridPoint, t2::kd.TimeGridPoint,
    greater=true)
    return G_int.GF[t1, t2, greater]
end

function (G_int::IncSplineImaginaryTimeGF)(t1::kd.BranchPoint, t2::kd.BranchPoint)
    return interpolate(G_int, t1, t2)
end

function extend!(G_int::IncSplineImaginaryTimeGF, val)
    grid = G_int.GF.grid
    norb = kd.norbitals(G_int)
    for k=1:norb, l=1:norb
        extend!(G_int.interpolants[k, l], val[k, l])
    end
    τ_0 = grid[1]
    τ = grid[length(G_int.interpolants[1, 1].data)]
    G_int.GF[τ, τ_0] = val
end

function interpolate(G_int::IncSplineImaginaryTimeGF{T, true},
                     t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = kd.get_ref(grid.contour, t1)
    ref2 = kd.get_ref(grid.contour, t2)

    return ref1 >= ref2 ? G_int.interpolants[1, 1](ref1 - ref2) :
                          Int(G_int.GF.ξ) * G_int.interpolants[1, 1](β + ref1 - ref2)
end

function interpolate(G_int::IncSplineImaginaryTimeGF{T, false},
                     t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = kd.get_ref(grid.contour, t1)
    ref2 = kd.get_ref(grid.contour, t2)

    norb = kd.norbitals(G_int.GF)
    if ref1 >= ref2
        return [G_int.interpolants[k, l](ref1 - ref2) for k=1:norb, l=1:norb]
    else
        return Int(G_int.GF.ξ) * [G_int.interpolants[k, l](β + ref1 - ref2)
                                  for k=1:norb, l=1:norb]
    end
end

end # module spline_gf
