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
