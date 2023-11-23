# [`QInchworm.sector_block_matrix`](@id QInchworm.sector_block_matrix)

```@meta
CurrentModule = QInchworm.sector_block_matrix
```
```@docs
sector_block_matrix
SectorBlockMatrix
operator_to_sector_block_matrix
Base.zeros(::Type{SectorBlockMatrix}, ::EDCore)
Base.zero(::SectorBlockMatrix)
Base.fill!(::SectorBlockMatrix, x)
LinearAlgebra.tr(::SectorBlockMatrix)
LinearAlgebra.norm(::SectorBlockMatrix, ::Real=2)
Statistics.var(::AbstractArray{SectorBlockMatrix}; ::Bool, mean)
Statistics.std(itr::AbstractArray{SectorBlockMatrix}; ::Bool, mean)
Base.isapprox(::SectorBlockMatrix, ::SectorBlockMatrix; ::Real)
```
