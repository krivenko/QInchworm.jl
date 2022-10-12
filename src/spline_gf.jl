module spline_gf

import Interpolations: BSplineInterpolation,
                       interpolate,
                       scale,
                       BSpline,
                       Cubic,
                       Line,
                       OnGrid

import Keldysh; kd = Keldysh;

import QInchworm.utility: get_ref

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

function SplineInterpolatedGF(GF::GFType) where {
        T <: Number, scalar, GFType <: kd.AbstractTimeGF{T, scalar}
    }
    norb = kd.norbitals(GF)
    interpolants = [make_interpolant(GF, k, l) for k=1:norb, l=1:norb]
    SplineInterpolatedGF{GFType, T, scalar}(GF, interpolants)
end

Base.eltype(::Type{<:SplineInterpolatedGF{GFType}}) where GFType = eltype(GFType)
Base.eltype(X::SplineInterpolatedGF) = eltype(typeof(X))

function Base.getproperty(G_int::SplineInterpolatedGF, p::Symbol)
    (p == :grid) ? G_int.GF.grid : getfield(G_int, p)
end

kd.norbitals(G_int::SplineInterpolatedGF) = kd.norbitals(G_int.GF)
kd.TimeDomain(G_int::SplineInterpolatedGF) = kd.TimeDomain(G_int.GF)

@inline function Base.getindex(G_int::SplineInterpolatedGF,
                               k, l,
                               t1::kd.TimeGridPoint, t2::kd.TimeGridPoint,
                               greater=true)
    G_int.GF[k, l, t1, t2, greater]
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
    v
end

@inline function Base.setindex!(G_int::SplineInterpolatedGF,
                                v,
                                t1::kd.TimeGridPoint, t2::kd.TimeGridPoint)
    G_int.GF[t1, t2] = v
    update_interpolants!(G_int)
    v
end

function (G_int::SplineInterpolatedGF)(t1::kd.BranchPoint, t2::kd.BranchPoint)
    return interpolate(G_int, t1, t2)
end

function update_interpolant!(G_int::SplineInterpolatedGF, k, l)
    G_int.interpolants[k, l] = make_interpolant(G_int.GF, k, l)
end

function update_interpolants!(G_int::SplineInterpolatedGF)
    norb = kd.norbitals(G_int)
    for k=1:norb, l=1:norb
        update_interpolant!(G_int, k, l)
    end
end

#
# Imaginary time GF
#

function make_interpolant(GF::kd.ImaginaryTimeGF{T, scalar}, k, l) where {T <: Number, scalar}
    grid = GF.grid
    knots = 0.:-imag(step(grid, kd.imaginary_branch)):grid.contour.β
    scale(interpolate(GF.mat.data[k,l,:], BSpline(Cubic(Line(OnGrid())))), knots)
end

function interpolate(G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, true}, T, true},
                     t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = get_ref(grid.contour, t1)
    ref2 = get_ref(grid.contour, t2)

    ref1 >= ref2 ? G_int.interpolants[1, 1](ref1 - ref2) :
                   Int(G_int.GF.ξ) * G_int.interpolants[1, 1](β + ref1 - ref2)
end

function interpolate(G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, false}, T, false},
    t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.GF.grid
    β = grid.contour.β
    ref1 = get_ref(grid.contour, t1)
    ref2 = get_ref(grid.contour, t2)

    norb = kd.norbitals(G_int.GF)
    if ref1 >= ref2
        [G_int.interpolants[k, l](ref1 - ref2) for k=1:norb, l=1:norb]
    else
        Int(G_int.GF.ξ) * [G_int.interpolants[k, l](β + ref1 - ref2) for k=1:norb, l=1:norb]
    end
end

end # module spline_gf
