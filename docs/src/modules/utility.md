# [`QInchworm.utility`](@id QInchworm.utility)

```@meta
CurrentModule = QInchworm.utility
```

## Interpolations.jl addon: Neumann boundary conditions for the cubic spline

```@docs
NeumannBC
Interpolations.prefiltering_system(::Type{T},
                                   ::Type{TC},
                                   ::Int,
                                   ::Interpolations.Cubic{BC}) where {
                                    T, TC, BC <: NeumannBC{Interpolations.OnGrid}}
```

## Quadratic spline that allows for incremental construction

```@docs
IncrementalSpline
extend!
```

## Sobol sequence including the initial point ``(0, 0, \ldots)``

```@docs
SobolSeqWith0
next!
arbitrary_skip!
```

## Lazy matrix product

```@docs
LazyMatrixProduct
Base.pushfirst!(::LazyMatrixProduct{T}, ::Matrix{T}) where {T <: Number}
Base.popfirst!(::LazyMatrixProduct{T}, ::Int) where {T <: Number}
eval!(::LazyMatrixProduct{T}) where {T <: Number}
```

## Serialization using `IOBuffer`

```@docs
iobuffer_serialize
iobuffer_deserialize
```

## Range partitioning utilities

```@docs
split_count
range_from_chunks_and_idx
```
