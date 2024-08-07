# QInchworm.jl

The package [QInchworm.jl](https://github.com/krivenko/QInchworm.jl)
is a Julia implementation of the quasi Monte Carlo variant[^1]
of the inchworm algorithm[^2] for solving impurity models with multiple
interacting fermions. Using quasi Monte Carlo, a ``1/N`` convergence rate with
the number of samples is achievable, which compares favorably to the
``1/\sqrt{N}`` convergence of the Monte Carlo methods.

Below, you can find an API reference of QInchworm.jl's modules.
Some parts of the API, such as handling of the atomic problem and of the pair
interactions/hybridization, depend on container types from
[Keldysh.jl](https://github.com/kleinhenz/Keldysh.jl) and exact diagonalization
tools of [KeldyshED.jl](https://github.com/krivenko/KeldyshED.jl).

There is also an [example page](@ref example) showing how to use QInchworm.jl
to solve a quantum impurity problem in thermodynamic equilibrium and to compute
two-point correlation functions for it.

[^1]: [Inchworm quasi Monte Carlo for quantum impurities.
       Hugo U. R. Strand, Joseph Kleinhenz and Igor Krivenko.
       arXiv:2310.16957.](https://arxiv.org/abs/2310.16957)
[^2]: [Taming the Dynamical Sign Problem in Real-Time Evolution of Quantum
       Many-Body Problems.
       Guy Cohen, Emanuel Gull, David R. Reichman, and Andrew J. Millis.
       Phys. Rev. Lett. 115, 266802 (2015).]
      (https://link.aps.org/doi/10.1103/PhysRevLett.115.266802)

### Public API

- [`QInchworm.expansion`](@ref api:QInchworm.expansion)
- [`QInchworm.inchworm`](@ref api:QInchworm.inchworm)
- [`QInchworm.randomization`](@ref api:QInchworm.randomization)
- [`QInchworm.ppgf`](@ref api:QInchworm.ppgf)
- [`QInchworm.spline_gf`](@ref api:QInchworm.spline_gf)
- [`QInchworm.utility`](@ref api:QInchworm.utility)

### Internals

- [`QInchworm.expansion`](@ref QInchworm.expansion)
- [`QInchworm.inchworm`](@ref QInchworm.inchworm)
- [`QInchworm.ppgf`](@ref QInchworm.ppgf)
- [`QInchworm.exact_atomic_ppgf`](@ref QInchworm.exact_atomic_ppgf)
- [`QInchworm.sector_block_matrix`](@ref QInchworm.sector_block_matrix)
- [`QInchworm.spline_gf`](@ref QInchworm.spline_gf)
- [`QInchworm.keldysh_dlr`](@ref QInchworm.keldysh_dlr)
- [`QInchworm.utility`](@ref QInchworm.utility)
- [`QInchworm.mpi`](@ref QInchworm.mpi)
- [`QInchworm.scrambled_sobol`](@ref QInchworm.scrambled_sobol)
- [`QInchworm.qmc_integrate`](@ref QInchworm.qmc_integrate)
- [`QInchworm.randomization`](@ref QInchworm.randomization)
- [`QInchworm.diagrammatics`](@ref QInchworm.diagrammatics)
- [`QInchworm.topology_eval`](@ref QInchworm.topology_eval)
