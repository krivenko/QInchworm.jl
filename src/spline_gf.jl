module spline_gf

import Interpolations: BSplineInterpolation,
                       interpolate,
                       scale,
                       BSpline,
                       Cubic,
                       Free,
                       OnGrid

import Keldysh; kd = Keldysh;

import QInchworm.utility: get_ref

"""
Wrapper around a Green's function object that allows for
fast cubic spline interpolation on the time grid.
"""
struct SplineInterpolatedGF{GFType, T, scalar} <: kd.AbstractTimeGF{T, scalar}
    "Wrapped Green's function"
    G::GFType
    "B-spline interpolants, one object per matrix element of G"
    interpolants
end

function SplineInterpolatedGF(G::GFType) where {
        T <: Number, scalar, GFType <: kd.AbstractTimeGF{T, scalar}
    }
    norb = kd.norbitals(G)
    interpolants = [make_interpolant(G, k, l) for k=1:norb, l=1:norb]
    SplineInterpolatedGF{GFType, T, scalar}(deepcopy(G), interpolants)
end

Base.eltype(::Type{<:SplineInterpolatedGF{GFType, T}}) where {GFType, T} = T
Base.eltype(X::SplineInterpolatedGF) = eltype(typeof(X))

function Base.getproperty(G_int::SplineInterpolatedGF, p::Symbol)
    (p == :grid) ? G_int.G.grid : getfield(G_int, p)
end

kd.norbitals(G_int::SplineInterpolatedGF) = kd.norbitals(G_int.G)
kd.TimeDomain(G_int::SplineInterpolatedGF) = kd.TimeDomain(G_int.G)

@inline function Base.getindex(G_int::SplineInterpolatedGF,
                               k, l,
                               t1::kd.TimeGridPoint, t2::kd.TimeGridPoint,
                               greater=true)
    Base.getindex(G_int.G, k, l, t1, t2, greater)
end

@inline function Base.getindex(G_int::SplineInterpolatedGF,
                               t1::kd.TimeGridPoint, t2::kd.TimeGridPoint,
                               greater=true)
    Base.getindex(G_int.G, t1, t2, greater)
end

@inline function Base.setindex!(G_int::SplineInterpolatedGF,
                                v,
                                k, l,
                                t1::kd.TimeGridPoint, t2::kd.TimeGridPoint)
    Base.setindex!(G_int.G, v, k, l, t1, t2)
    G_int.interpolants[k, l] = make_interpolant(G_int.G, k, l)
    v
end

@inline function Base.setindex!(G_int::SplineInterpolatedGF,
                                v,
                                t1::kd.TimeGridPoint, t2::kd.TimeGridPoint)
    Base.setindex!(G_int.G, v, t1, t2)
    norb = kd.norbitals(G_int.G)
    for k=1:norb, l=1:norb
        G_int.interpolants[k, l] = make_interpolant(G_int.G, k, l)
    end
    v
end

function (G_int::SplineInterpolatedGF)(t1::kd.BranchPoint, t2::kd.BranchPoint)
    return interpolate(G_int, t1, t2)
end

#
# Imaginary time GF
#

function make_interpolant(G::kd.ImaginaryTimeGF{T, scalar}, k, l) where {T <: Number, scalar}
    grid = G.grid
    knots = 0.:-imag(step(grid, kd.imaginary_branch)):grid.contour.β
    scale(interpolate(G.mat.data[k,l,:], BSpline(Cubic(Free(OnGrid())))), knots)
end

function interpolate(G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, true}, T, true},
                     t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.G.grid
    β = grid.contour.β
    ref1 = get_ref(grid.contour, t1)
    ref2 = get_ref(grid.contour, t2)

    ref1 >= ref2 ? G_int.interpolants[1, 1](ref1 - ref2) :
                   Int(G_int.G.ξ) * G_int.interpolants[1, 1](β + ref1 - ref2)
end

function interpolate(G_int::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, false}, T, false},
    t1::kd.BranchPoint, t2::kd.BranchPoint) where T
    grid = G_int.G.grid
    β = grid.contour.β
    ref1 = get_ref(grid.contour, t1)
    ref2 = get_ref(grid.contour, t2)

    norb = kd.norbitals(G_int.G)
    if ref1 >= ref2
        [G_int.interpolants[k, l](ref1 - ref2) for k=1:norb, l=1:norb]
    else
        Int(G_int.G.ξ) * [G_int.interpolants[k, l](β + ref1 - ref2) for k=1:norb, l=1:norb]
    end
end

end # module spline_gf
