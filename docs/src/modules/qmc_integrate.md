# [`QInchworm.qmc_integrate`](@id QInchworm.qmc_integrate)

```@meta
CurrentModule = QInchworm.qmc_integrate
```
```@docs
qmc_integrate
```

## Basic quasi Monte Carlo integration routines

```@docs
qmc_integral
qmc_integral_n_samples
```

## Integration domain transformations

```@docs
AbstractDomainTransform
Base.ndims(::AbstractDomainTransform)
ExpModelFunctionTransform
ExpModelFunctionTransform(::Integer, ::kd.AbstractContour, ::kd.BranchPoint, ::Real)
RootTransform
RootTransform(::Integer, ::kd.AbstractContour, ::kd.BranchPoint, ::kd.BranchPoint)
SortTransform
SortTransform(::Integer, ::kd.AbstractContour, ::kd.BranchPoint, ::kd.BranchPoint)
DoubleSimplexRootTransform
DoubleSimplexRootTransform(::Int,
                           ::Int,
                           ::kd.AbstractContour,
                           ::kd.BranchPoint,
                           ::kd.BranchPoint,
                           ::kd.BranchPoint)
make_trans_f
make_jacobian_f
```

## Contour integration routines

```@docs
contour_integral
contour_integral_n_samples
branch_direction
contour_function_return_type
```