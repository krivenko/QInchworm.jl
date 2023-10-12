TODO list for QInchworm
=======================

* Compute the partition function Z and the Gibbs free energy \Omega
  of the expansion (relative to the atomic system) before normalizing.

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

* Direction coefficients calculated in `qmc_*_integral_*()` are ambiguous at the branch
  boundaries. This is going to become a problem once we generalize the solver to
  the full Keldysh contour.