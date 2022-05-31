```@meta
CurrentModule = QInchworm.configuration
```

# [`QInchworm.configuration`](@id QInchworm.configuration)

The `configuration` module is a framework for representing generic pseudo particle diagrams
with a back-bone of pseudo particle propagators and insertions of pseudo particle interactions
at fixed times.

There is also functionality to represent Inch-Worm diagrams with a fixed "inch" time where
the back bone propagator switches from bold to bare.

The building blocks common for all kinds of diagrams are contained in the [`Expansion`](@ref) struct.
Given an [`Expansion`](@ref) a particular diagram with fixed times can be represented by the [`Configuration`](@ref) struct.

Given an [`Expansion`](@ref) and a [`Configuration`](@ref) the [`eval`](@ref) function can be used to evaluate the value
of the "configuration"/diagram.


## Pseudo particle interactions

In pseudo particle strong coupling expansions the notion "hybridization insertions" can be seen as a retarded interaction
between the pseudo particles.

One such pseudo particle interaction is defined by a scalar hybridization function Green's function and an initial and final many-body operator.

Thus, a pseudo particle `Expansion` is defined, in part, by the list of all possible pseudo particle interactions in the system,
where each pseudo particle interaction is represented using the struct [`InteractionPair`](@ref).


## Pseudo-particle symmetry sectors

The pseudo-particle propagator and the many-body operators are represented as block matrices in the local Hilbert space.
The blocking is performed by `KeldyshED.jl` based on the symmetries of the local Hamiltonian.

To represent many-body operators the [`SectorBlockMatrix`](@ref) type is used.


## Module Index

```@index
Modules = [configuration]
```

## Module API Documentation

```@autodocs
Modules = [configuration]
```
