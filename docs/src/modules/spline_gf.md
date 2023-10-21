```@meta
CurrentModule = QInchworm
```

# [`QInchworm.spline_gf`](@id QInchworm.spline_gf)

```@autodocs
Modules = [QInchworm.spline_gf]
Order   = [:module]
```

## Public API

```@autodocs
Modules = [QInchworm.spline_gf]
Order = [:type, :function]
Private = false
```

## Internals

```@meta
CurrentModule = QInchworm.spline_gf
```

```@docs
make_interpolant(::kd.ImaginaryTimeGF{T, scalar}, k, l, ::kd.TimeGridPoint) where {
    T <: Number, scalar}
update_interpolant!(::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar},
                    k, l; ::kd.TimeGridPoint) where {T <: Number, scalar}
update_interpolants!(::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar};
                     ::kd.TimeGridPoint) where {T <: Number, scalar}
interpolate(::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, true}, T, true},
            ::kd.BranchPoint, ::kd.BranchPoint) where T
interpolate(::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, false}, T, false},
            ::kd.BranchPoint, ::kd.BranchPoint) where T
IncSplineImaginaryTimeGF
IncSplineImaginaryTimeGF(::kd.ImaginaryTimeGF{T, true}, ::T) where {T <: Number}
IncSplineImaginaryTimeGF(::kd.ImaginaryTimeGF{T, false}, ::Matrix{T}) where {T <: Number}
make_inc_interpolant(::kd.ImaginaryTimeGF{T, scalar}, k, l, ::T) where {T <: Number, scalar}
Base.zero(::IncSplineImaginaryTimeGF{T, false}) where {T <: Number}
Base.zero(::IncSplineImaginaryTimeGF{T, true}) where {T <: Number}
extend!(::IncSplineImaginaryTimeGF, val)
interpolate(::IncSplineImaginaryTimeGF{T, true}, ::kd.BranchPoint, ::kd.BranchPoint) where T
interpolate(::IncSplineImaginaryTimeGF{T, false}, ::kd.BranchPoint, ::kd.BranchPoint) where T
```