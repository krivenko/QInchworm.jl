# [`QInchworm.topology_eval`](@id QInchworm.topology_eval)

```@meta
CurrentModule = QInchworm.topology_eval
```
```@docs
topology_eval
NodeKind
Node
FixedNode
IdentityNode
InchNode
OperatorNode
TopologyEvaluator
TopologyEvaluator(::Expansion, ::Int, ::Dict{Int, FixedNode}; ::TimerOutput)
eval(::Topology, ::Vector{kd.BranchPoint})
eval(::Vector{Topology}, ::Vector{kd.BranchPoint})
_traverse_configuration_tree!
```
