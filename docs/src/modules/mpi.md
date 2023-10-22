# [`QInchworm.mpi`](@id QInchworm.mpi)

```@meta
CurrentModule = QInchworm.mpi
```
```@docs
mpi
ismaster
rank_sub_range
all_gather(::Vector{T}; ::MPI.Comm) where T
all_reduce!(::SectorBlockMatrix, op; ::MPI.Comm)
```