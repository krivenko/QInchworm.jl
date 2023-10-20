```@meta
CurrentModule = QInchworm
```

# [`QInchworm.utility`](@id QInchworm.utility)

```@autodocs
Modules = [QInchworm.utility]
Order   = [:module]
```

## Public API

```@autodocs
Modules = [QInchworm.utility]
Order   = [:type, :function]
Private = false
```

## Internals

```@meta
CurrentModule = QInchworm.utility
```

```@docs
NeumannBC
Interpolations.prefiltering_system(::Type{T},
                                   ::Type{TC},
                                   ::Int,
                                   ::Interpolations.Cubic{BC}) where {
                                    T, TC, BC <: NeumannBC{Interpolations.OnGrid}}
IncrementalSpline
extend!
SobolSeqWith0
next!
arbitrary_skip!
split_count
range_from_chunks_and_idx
iobuffer_serialize
iobuffer_deserialize
LazyMatrixProduct
Base.pushfirst!(::LazyMatrixProduct{T}, ::Matrix{T}) where {T <: Number}
Base.popfirst!(::LazyMatrixProduct{T}, ::Int) where {T <: Number}
eval!(::LazyMatrixProduct{T}) where {T <: Number}
```
