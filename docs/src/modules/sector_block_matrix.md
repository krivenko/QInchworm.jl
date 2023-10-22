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
Base.isapprox(::SectorBlockMatrix, ::SectorBlockMatrix; atol::Real)
```
