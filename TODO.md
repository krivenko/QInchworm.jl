TODO list for QInchworm
=======================

* Rename `BetterSobolSeq` -> `SobolSeqWith0`.
* Consider renaming `InchwormOrderData` -> `ExpansionOrderData`.
  This data structure is now used in both inchworm propagation
  (PPGF accumulation) and GF accumulation.
* Consider renaming `d_bare`, `d_bold` and `k_attached`.
  I wonder if terms 'before' and 'after' (or something similar)
  would make more sense in the light of the bold propagators
  being used on both sides of t_w.
* `SectorBlockMatrix` deserves its own `.jl` file.