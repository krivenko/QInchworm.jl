# [`QInchworm.topology_eval`](@id QInchworm.topology_eval)

```@meta
CurrentModule = QInchworm.topology_eval
```
```@docs
topology_eval
NodeKind
Node
FixedNode
PairNode
IdentityNode
InchNode
OperatorNode
EvolvingSectorBlockMatrix
EvolvingSectorBlockMatrix(::SectorBlockMatrix)
get_finite_time
update_finite_time!
TopologyEvaluator
TopologyEvaluator(::Expansion, ::Int, ::Bool, ::Dict{Int, FixedNode}; ::TimerOutput)
TopologyEvaluator(::Topology, ::Vector{kd.BranchPoint})
TopologyEvaluator(::Vector{Topology}, ::Vector{kd.BranchPoint})
_traverse_configuration_tree!
```
