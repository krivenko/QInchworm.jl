# QInchworm.jl
#
# Copyright (C) 2021-2025 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
Spline-interpolated Green's function containers.

# Exports
$(EXPORTS)
"""
module spline_gf

using DocStringExtensions

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

export SplineInterpolatedGF
export ph_conj

#
# SplineInterpolatedGF
#

"""
    $(TYPEDEF)

Wrapper around a Green's function object that allows for fast cubic spline interpolation on
the time grid.

The wrapper supports square bracket access to the wrapped object, direct access to the
`grid` property, `eltype()`, `Keldysh.norbitals()` and `Keldysh.TimeDomain()`. Evaluation at
an arbitrary contour time point (via operator `()`) is carried out by a stored set of
pre-computed B-spline interpolants.

# Fields
$(TYPEDFIELDS)
"""
struct SplineInterpolatedGF{GFType, T, scalar} <: kd.AbstractTimeGF{T, scalar}
    "Wrapped Green's function"
    GF::GFType
    "B-spline interpolants, one object per matrix element of G"
    interpolants
end

"""
    $(TYPEDSIGNATURES)

Make a [`SplineInterpolatedGF`](@ref) wrapper around `GF` and compute interpolants of its
data from the start of the grid up to `τ_max`. By default, the entire data array is used.
"""
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

"""
    $(TYPEDSIGNATURES)

Update the interpolant stored in `G_int` and corresponding to its matrix indices `k`, `l`.
The updated interpolant interpolates data points of `G_int.GF` from the start of the grid
up to `τ_max`. By default, the entire data array is used.
"""
function update_interpolant!(
    G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar},
    k, l;
    τ_max::kd.TimeGridPoint = G_int.GF.grid[end]) where {T <: Number, scalar}
    G_int.interpolants[k, l] = make_interpolant(G_int.GF, k, l, τ_max)
end

"""
    $(TYPEDSIGNATURES)

Update all interpolants stored in `G_int`. The updated interpolants interpolate data points
of `G_int.GF` from the start of the grid up to `τ_max`. By default, the entire data array is
used.
"""
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

"""
    $(TYPEDSIGNATURES)

Make a cubic B-spline interpolant from `GF`'s data corresponding to its matrix indices
`k`, `l`. Data points from the start of the grid up to `τ_max` are used.
"""
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

"""
    $(TYPEDSIGNATURES)

Evaluate the spline-interpolated Green's function `G_int` at the contour time arguments
`t1`, `t2`.
"""
function interpolate(G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, true}, T, true},
                     t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = kd.get_ref(grid.contour, t1)
    ref2 = kd.get_ref(grid.contour, t2)

    return ref1 >= ref2 ? G_int.interpolants[1, 1](ref1 - ref2) :
                          Int(G_int.GF.ξ) * G_int.interpolants[1, 1](β + ref1 - ref2)
end

"""
    $(TYPEDSIGNATURES)

Evaluate the spline-interpolated Green's function `G_int` at the contour time arguments
`t1`, `t2`.
"""
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

"""
    $(TYPEDSIGNATURES)

Given a spline-interpolated scalar-valued imaginary time Green's function ``g(\\tau)``,
return its particle-hole conjugate ``g(\\beta-\\tau)``.
"""
function ph_conj(G_int::SplineInterpolatedGF{GFType, T, true}) where {
        T <: Number, GFType <: kd.ImaginaryTimeGF{T, true}
    }
    return SplineInterpolatedGF{GFType, T, true}(ph_conj(G_int.GF))
end

#
# IncSplineImaginaryTimeGF
#

"""
    $(TYPEDEF)

Wrapper around an imaginary time Green's function object that supports interpolation based
on the [`IncrementalSpline`](@ref).

The wrapper supports square bracket access to the wrapped object, direct access to the
`grid` property, `eltype()`, `Keldysh.norbitals()` and `Keldysh.TimeDomain()`. Evaluation at
an arbitrary imaginary time point (via operator `()`) is carried out by a stored set of
pre-computed [`IncrementalSpline`](@ref) interpolants.

# Fields
$(TYPEDFIELDS)
"""
struct IncSplineImaginaryTimeGF{T, scalar} <: kd.AbstractTimeGF{T, scalar}
    "Wrapped Green's function"
    GF::kd.ImaginaryTimeGF{T, scalar}
    "Incremental spline interpolants, one object per matrix element of GF"
    interpolants

    @doc """
        $(TYPEDSIGNATURES)

    Make a [`IncSplineImaginaryTimeGF`](@ref) wrapper around a scalar-valued `GF` and
    initialize incremental interpolants of its data. `derivative_at_0` is imaginary time
    derivative of `GF` at ``\\tau=0`` needed to compute the first segment of the
    interpolants.
    """
    function IncSplineImaginaryTimeGF(GF::kd.ImaginaryTimeGF{T, true},
                                      derivative_at_0::T) where {T <: Number}
        interpolants = [make_inc_interpolant(GF, k, l, derivative_at_0)]
        return new{T, true}(GF, interpolants)
    end
    @doc """
        $(TYPEDSIGNATURES)

    Make a [`IncSplineImaginaryTimeGF`](@ref) wrapper around a matrix-valued `GF` and
    initialize incremental interpolants of its data. `derivative_at_0` is imaginary time
    derivative of `GF` at ``\\tau=0`` needed to compute the first segment of the
    interpolants.
    """
    function IncSplineImaginaryTimeGF(GF::kd.ImaginaryTimeGF{T, false},
                                      derivative_at_0::Matrix{T}) where {T <: Number}
        norb = kd.norbitals(GF)
        interpolants = [make_inc_interpolant(GF, k, l, derivative_at_0[k, l])
                        for k=1:norb, l=1:norb]
        return new{T, false}(GF, interpolants)
    end
end

"""
    $(TYPEDSIGNATURES)

Make an incremental spline interpolant from `GF`'s data and corresponding to its matrix
indices `k`, `l`. `derivative_at_0` is imaginary time derivative of `GF[k, l]` at
``\\tau=0`` needed to compute the first segment of the interpolant.
"""
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

"""
    $(TYPEDSIGNATURES)

Make a zero matrix-valued [`IncSplineImaginaryTimeGF`](@ref) object similar to `G_int`.
"""
function Base.zero(G_int::IncSplineImaginaryTimeGF{T, false}) where {T <: Number}
    norb = kd.norbitals(G_int.GF)
    return IncSplineImaginaryTimeGF(kd.zero(G_int.GF), zeros(T, (norb, norb)))
end

"""
    $(TYPEDSIGNATURES)

Make a zero scalar-valued [`IncSplineImaginaryTimeGF`](@ref) object similar to `G_int`.
"""
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

"""
    $(TYPEDSIGNATURES)

Extend the underlying [`IncrementalSpline`](@ref) objects stored in `G_int` with a value
`val`.
"""
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

"""
    $(TYPEDSIGNATURES)

Evaluate the spline-interpolated scalar-valued Green's function `G_int` at the imaginary
time arguments `τ1`, `τ2`.
"""
function interpolate(G_int::IncSplineImaginaryTimeGF{T, true},
                     τ1::kd.BranchPoint, τ2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = kd.get_ref(grid.contour, τ1)
    ref2 = kd.get_ref(grid.contour, τ2)

    return ref1 >= ref2 ? G_int.interpolants[1, 1](ref1 - ref2) :
                          Int(G_int.GF.ξ) * G_int.interpolants[1, 1](β + ref1 - ref2)
end

"""
    $(TYPEDSIGNATURES)

Evaluate the spline-interpolated matrix-valued Green's function `G_int` at the imaginary
time arguments `τ1`, `τ2`.
"""
function interpolate(G_int::IncSplineImaginaryTimeGF{T, false},
                     τ1::kd.BranchPoint, τ2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = kd.get_ref(grid.contour, τ1)
    ref2 = kd.get_ref(grid.contour, τ2)

    norb = kd.norbitals(G_int.GF)
    if ref1 >= ref2
        return [G_int.interpolants[k, l](ref1 - ref2) for k=1:norb, l=1:norb]
    else
        return Int(G_int.GF.ξ) * [G_int.interpolants[k, l](β + ref1 - ref2)
                                  for k=1:norb, l=1:norb]
    end
end

end # module spline_gf
