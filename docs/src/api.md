# Public API

```@docs
QInchworm
```

## [`QInchworm.expansion`](@id api:QInchworm.expansion)

```@meta
CurrentModule = QInchworm.expansion
```
```@docs
expansion
```
```@docs
Expansion
Expansion(::ked.EDCore,
          ::kd.AbstractTimeGrid,
          ::Vector{InteractionPair{ScalarGF}};
          ::Vector{Tuple{Operator, Operator}},
          interpolate_ppgf) where ScalarGF
Expansion(::op.OperatorExpr,
          ::ked.SetOfIndices,
          ::kd.ImaginaryTimeGrid;
          ::kd.ImaginaryTimeGF{ComplexF64, false},
          ::Union{Keldysh.ImaginaryTimeGF{Float64, false}, Nothing},
          ::Vector{Tuple{Operator, Operator}},
          interpolate_ppgf)
InteractionPair
add_corr_operators!
```

## [`QInchworm.inchworm`](@id api:QInchworm.inchworm)

```@meta
CurrentModule = QInchworm.inchworm
```
```@docs
inchworm
```
```@docs
inchworm!
correlator_2p
```

## [`QInchworm.randomization`](@id api:QInchworm.randomization)

```@meta
CurrentModule = QInchworm.randomization
```
```@docs
randomization
```
```@docs
RandomizationParams
```

## [`QInchworm.ppgf`](@id api:QInchworm.ppgf)

```@meta
CurrentModule = QInchworm.ppgf
```
```@docs
ppgf
```
```@docs
FullTimePPGF
ImaginaryTimePPGF
atomic_ppgf
partition_function
density_matrix
normalize!
```

## [`QInchworm.spline_gf`](@id api:QInchworm.spline_gf)

```@meta
CurrentModule = QInchworm.spline_gf
```
```@docs
spline_gf
```
```@docs
SplineInterpolatedGF
SplineInterpolatedGF(::GFType; ::kd.TimeGridPoint) where {
        T <: Number, scalar, GFType <: kd.AbstractTimeGF{T, scalar}}
```

## [`QInchworm.utility`](@id api:QInchworm.utility)

```@meta
CurrentModule = QInchworm.utility
```
```@docs
utility
```
```@docs
ph_conj
```
