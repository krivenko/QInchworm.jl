TODO list for QInchworm
=======================

* How about not storing time points in configurations?
  This way we don't have to update them before calling eval().
  Instead, we could pass a list of time points directly to eval().
  In a Markov chain algorithm one has only one configuration
  that keeps track of the time points so that they can be used
  to propose a new configuration. In our code, however, that
  information is superfluous as we always generate a whole
  list of time points independently from the previously generated
  lists.

* Compute the partition function Z and the Gibbs free energy \Omega
  of the expansion (relative to the atomic system) before normalizing.

* One performance bottleneck is the generation of the operator matrices
  consider storing the matrix representations in each configuration.

* Consider using [Logging](https://docs.julialang.org/en/v1/stdlib/Logging/)
  instead of `println()` / `@show`.

* Rethink parallelization strategy in order to handle high order expansions

  - Our current approach is to pre generate all non-zero configurations
    storing a copy of all of them on every mpi node.

    For a two fermion problem we currently run out of memory at order 8.

  - The parallell calculation is solely over the quasi Monte Carlo evaluations
    which is done by all mpi ranks in tandem for each and every diagram+configration.

  - To get to order 8 and beyond, consider designing a distributed generation of
    configurations with separate quasi Monte Carlo evaluation of subset of diagrams?

    The distribution of configurations could still be done order-by-order
    inorder to avoid dealing with load re-distribution.
