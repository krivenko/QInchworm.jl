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

## Lazy matrix product

```@docs
LazyMatrixProduct
Base.pushfirst!(::LazyMatrixProduct{T}, ::Matrix{T}) where {T <: Number}
Base.popfirst!(::LazyMatrixProduct{T}, ::Int) where {T <: Number}
eval!(::LazyMatrixProduct{T}) where {T <: Number}
```

## Random sequence

```@docs
RandomSeq
seed!
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
