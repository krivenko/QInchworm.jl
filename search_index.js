var documenterSearchIndex = {"docs":
[{"location":"modules/diagrammatics/","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics","text":"CurrentModule = QInchworm","category":"page"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics","text":"","category":"section"},{"location":"modules/diagrammatics/","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics","text":"Modules = [QInchworm.diagrammatics]","category":"page"},{"location":"modules/diagrammatics/","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics","text":"Modules = [QInchworm.diagrammatics]","category":"page"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.Diagram","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.Diagram","text":"Diagram with a topology and tuple of pseudo particle interaction pair indices\n\n\n\n\n\n","category":"type"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.Topology","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.Topology","text":"struct Topology\n\nDatatype for diagram topology. A topology of order n consists of a partition of the ordered set s = 12n into n pairs \\{(x(1), x(2)), ..., (x(2n-1), x(2n))\\} where x is a permutation of s. Diagrammatically a topology can be thought of as a set of arcs connecting vertices located at 12n. The parity of the topology is the sign of the permutation x.\n\norder::Int64\npairs::Vector{Pair{Int64, Int64}}\nparity::Int64\n\n\n\n\n\n","category":"type"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.count_doubly_k_connected-Tuple{Vector{Pair{Int64, Int64}}, Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.count_doubly_k_connected","text":"count_doubly_k_connected(\n    pairs::Vector{Pair{Int64, Int64}},\n    k::Int64\n) -> Int64\n\n\nGiven a vector of pairs, count the doubly k-connected ones.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.generate_topologies-Tuple{Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.generate_topologies","text":"generate_topologies(\n    n::Int64\n) -> Vector{QInchworm.diagrammatics.Topology}\n\n\nReturn topologies of order n, efficiently computing the permutation sign for each.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.generate_topologies_impl-Tuple{QInchworm.diagrammatics.Topology, Vector{Int64}}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.generate_topologies_impl","text":"generate_topologies_impl(\n    topology_partial::QInchworm.diagrammatics.Topology,\n    unpaired::Vector{Int64}\n) -> Vector{QInchworm.diagrammatics.Topology}\n\n\nGiven a partial topology and a vector of unpaired vertices, return a vector of complete topologies, efficiently computing the permutation sign.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.is_doubly_k_connected-Tuple{Pair{Int64, Int64}, Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.is_doubly_k_connected","text":"is_doubly_k_connected(\n    p::Pair{Int64, Int64},\n    k::Int64\n) -> Bool\n\n\nReturns true if a given pair has one index <= k and the other index > k.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.is_doubly_k_connected-Tuple{QInchworm.diagrammatics.Topology, Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.is_doubly_k_connected","text":"is_doubly_k_connected(\n    t::QInchworm.diagrammatics.Topology,\n    k::Int64\n) -> Bool\n\n\nGiven a topology, check if every connected component of the graph induced by crossings between the arcs contains a pair with one index <= k and the other index > k\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.is_k_connected-Tuple{QInchworm.diagrammatics.Topology, Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.is_k_connected","text":"is_k_connected(\n    t::QInchworm.diagrammatics.Topology,\n    k::Int64\n) -> Bool\n\n\nGiven a topology, check if every connected component of the graph induced by crossings between the arcs contains a pair with index <= k\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.iscrossing-Tuple{Pair{Int64, Int64}, Pair{Int64, Int64}}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.iscrossing","text":"iscrossing(\n    p1::Pair{Int64, Int64},\n    p2::Pair{Int64, Int64}\n) -> Bool\n\n\nReturns true if two arcs cross\n\nLet p_1 = (a b), p_2 = (x y) represent two arcs, where without loss of generality we assume a  b and x  y. Now consider the order the points a b x y. The orderings abxy, axyb, xyab are all non-crossing while axby and xayb are crossing.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.n_crossings-Tuple{QInchworm.diagrammatics.Topology}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.n_crossings","text":"n_crossings(top::QInchworm.diagrammatics.Topology) -> Int64\n\n\nReturns the number of crossing arcs in a topology.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.pair_partitions-Tuple{Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.pair_partitions","text":"pair_partitions(\n    n::Int64\n) -> Vector{Vector{Pair{Int64, Int64}}}\n\n\nReturn all partitions of 12n into n pairs.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.pair_partitions-Tuple{Vector{Int64}}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.pair_partitions","text":"pair_partitions(\n    vertices::Vector{Int64}\n) -> Vector{Vector{Pair{Int64, Int64}}}\n\n\nReturn all partitions of the vertices into pairs.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.pair_partitions-Tuple{Vector{Pair{Int64, Int64}}, Vector{Int64}}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.pair_partitions","text":"pair_partitions(\n    pairs::Vector{Pair{Int64, Int64}},\n    unpaired::Vector{Int64}\n) -> Vector{Vector{Pair{Int64, Int64}}}\n\n\nGiven a vector of pairs representing a partial partition of the vertices and a vector of unpaired vertices, return a vector of complete partitions.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.parity-Tuple{QInchworm.diagrammatics.Topology}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.parity","text":"parity(top::QInchworm.diagrammatics.Topology) -> Int64\n\n\nReturns the parity of the permutation matrix of the topolgy.\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.pop_pair-Tuple{Vector, Any, Any}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.pop_pair","text":"pop_pair(v::Vector, i, j) -> Tuple{Pair, Any}\n\n\nreturns the pair v[i] => v[j] and a copy of the vector v with elements i,j removed\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.split_doubly_k_connected-Tuple{Vector{Pair{Int64, Int64}}, Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.split_doubly_k_connected","text":"split_doubly_k_connected(\n    pairs::Vector{Pair{Int64, Int64}},\n    k::Int64\n) -> Tuple{Vector{Pair{Int64, Int64}}, Vector{Pair{Int64, Int64}}}\n\n\nGiven a vector of pairs, split it into a 'connected' set containing pairs with one index <= k and the other index > k and a disconnected set containing the rest\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.split_k_connected-Tuple{Vector{Pair{Int64, Int64}}, Int64}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.split_k_connected","text":"split_k_connected(\n    pairs::Vector{Pair{Int64, Int64}},\n    k::Int64\n) -> Tuple{Vector{Pair{Int64, Int64}}, Vector{Pair{Int64, Int64}}}\n\n\nGiven a vector of pairs, split it into a 'connected' set containing pairs with an index <= k and a disconnected set containing the rest\n\n\n\n\n\n","category":"method"},{"location":"modules/diagrammatics/#QInchworm.diagrammatics.traverse_crossing_graph_dfs!-Tuple{Vector{Pair{Int64, Int64}}, Vector{Pair{Int64, Int64}}}","page":"QInchworm.diagrammatics","title":"QInchworm.diagrammatics.traverse_crossing_graph_dfs!","text":"traverse_crossing_graph_dfs!(\n    connected::Vector{Pair{Int64, Int64}},\n    disconnected::Vector{Pair{Int64, Int64}}\n)\n\n\nGiven a vector of 'connected' arcs and a vector of 'disconnected' arcs recursively add disconnected to connected if they cross with any connected. This is done by traversing the crossing graph using depth first search.\n\n\n\n\n\n","category":"method"},{"location":"modules/ppgf/","page":"QInchworm.ppgf","title":"QInchworm.ppgf","text":"CurrentModule = QInchworm","category":"page"},{"location":"modules/ppgf/#QInchworm.ppgf","page":"QInchworm.ppgf","title":"QInchworm.ppgf","text":"","category":"section"},{"location":"modules/ppgf/","page":"QInchworm.ppgf","title":"QInchworm.ppgf","text":"Modules = [QInchworm.ppgf]","category":"page"},{"location":"modules/ppgf/","page":"QInchworm.ppgf","title":"QInchworm.ppgf","text":"Modules = [QInchworm.ppgf]","category":"page"},{"location":"modules/ppgf/#QInchworm.ppgf.atomic_ppgf-Tuple{Keldysh.FullTimeGrid, KeldyshED.EDCore}","page":"QInchworm.ppgf","title":"QInchworm.ppgf.atomic_ppgf","text":"Compute atomic pseudo-particle Green's function on the time grid for a time-independent problem defined by the EDCore instance.\n\n\n\n\n\n","category":"method"},{"location":"modules/ppgf/#QInchworm.ppgf.first_order_spgf-Tuple{Vector{Keldysh.GenericTimeGF{ComplexF64, false, Keldysh.FullTimeGrid}}, KeldyshED.EDCore, Any, Any}","page":"QInchworm.ppgf","title":"QInchworm.ppgf.first_order_spgf","text":"Compute the first order pseudo-particle diagram contribution to the single-particle Green's function g_{o1, o2}(z, z')\n\n\n\n\n\n","category":"method"},{"location":"modules/ppgf/#QInchworm.ppgf.operator_matrix_representation-Union{Tuple{S}, Tuple{KeldyshED.Operators.OperatorExpr{S}, KeldyshED.EDCore}} where S<:Number","page":"QInchworm.ppgf","title":"QInchworm.ppgf.operator_matrix_representation","text":"Get matrix representation of operator expression in each sector\n\nNB! Requires that the operator expression does not mix symmetry sectors\n\n\n\n\n\n","category":"method"},{"location":"modules/ppgf/#QInchworm.ppgf.operator_product-Tuple{KeldyshED.EDCore, Any, Integer, Any, Any, Any}","page":"QInchworm.ppgf","title":"QInchworm.ppgf.operator_product","text":"operator_product(...)\n\nEvaluate a product of vertices at different contour times z_i with the pseudo-particle Green's function sandwiched in between.\n\nvertices is a contour-time ordered list of triples (z_i, c_i, o_i) were:   z_i is a contour time,   c_i is +1/-1 for creation/annihilation operator respectively, and   o_i is a spin-orbital index\n\n\n\n\n\n","category":"method"},{"location":"modules/ppgf/#QInchworm.ppgf.set_matsubara!-Tuple{Keldysh.GenericTimeGF{T, scalar, Keldysh.FullTimeGrid} where {T, scalar}, Any, Any}","page":"QInchworm.ppgf","title":"QInchworm.ppgf.set_matsubara!","text":"Set all time translation invariant values of the Matsubara branch\n\n\n\n\n\n","category":"method"},{"location":"modules/ppgf/#QInchworm.ppgf.set_ppgf_symmetric!-Tuple{Vector{Keldysh.GenericTimeGF{ComplexF64, false, Keldysh.FullTimeGrid}}, Vararg{Any, 4}}","page":"QInchworm.ppgf","title":"QInchworm.ppgf.set_ppgf_symmetric!","text":"Set real-time ppgf symmetry connected time pairs\n\nNB! times has to be in the inching region with z2 ∈ backward_branch.\n\n\n\n\n\n","category":"method"},{"location":"modules/qmc_integrate/","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate","text":"CurrentModule = QInchworm","category":"page"},{"location":"modules/qmc_integrate/#QInchworm.qmc_integrate","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate","text":"","category":"section"},{"location":"modules/qmc_integrate/","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate","text":"Modules = [QInchworm.qmc_integrate]","category":"page"},{"location":"modules/qmc_integrate/","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate","text":"Modules = [QInchworm.qmc_integrate]","category":"page"},{"location":"modules/qmc_integrate/#QInchworm.qmc_integrate.contour_function_return_type-Tuple{Function}","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate.contour_function_return_type","text":"Detect the return type of a function applied to a vector of branch points.\n\n\n\n\n\n","category":"method"},{"location":"modules/qmc_integrate/#QInchworm.qmc_integrate.exp_p_norm-Tuple{Real, Int64}","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate.exp_p_norm","text":"Normalization of the model function p_d(u)\n\n\n\n\n\n","category":"method"},{"location":"modules/qmc_integrate/#QInchworm.qmc_integrate.make_exp_model_function-Tuple{Keldysh.AbstractContour, Keldysh.BranchPoint, Real, Int64}","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate.make_exp_model_function","text":"Make p_d(u) from the exponential h(v).\n\n\n\n\n\n","category":"method"},{"location":"modules/qmc_integrate/#QInchworm.qmc_integrate.make_model_function-Tuple{Keldysh.AbstractContour, Keldysh.BranchPoint, Vector}","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate.make_model_function","text":"Make the model function p_d(u) out of h_i(v).\n\n\n\n\n\n","category":"method"},{"location":"modules/qmc_integrate/#QInchworm.qmc_integrate.qmc_integral","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate.qmc_integral","text":"Quasi Monte Carlo integration with warping.\n\n`f`      Integrand.\n`init`   Initial value of the integral.\n`p`      Positive model function p_n(u).\n`p_norm` Integral of p_n(u) over the u-domain.\n`trans`  Transformation from x ∈ [0,1]^d onto the u-domain.\n`seq`    Quasi-random sequence generator.\n`N`      The number of points taken from the quasi-random sequence.\n\n\n\n\n\n","category":"function"},{"location":"modules/qmc_integrate/#QInchworm.qmc_integrate.qmc_integral_n_samples","page":"QInchworm.qmc_integrate","title":"QInchworm.qmc_integrate.qmc_integral_n_samples","text":"Quasi Monte Carlo integration with warping.\n\nThis function takes a specified number of valid samples of the integrand.\n`nothing` returned by the integrand does not count towards this number.\n\n`f`         Integrand.\n`init`      Initial value of the integral.\n`p`         Positive model function p_n(u).\n`p_norm`    Integral of p_n(u) over the u-domain.\n`trans`     Transformation from x ∈ [0,1]^d onto the u-domain.\n`seq`       Quasi-random sequence generator.\n`N_samples` The number of taken samples.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = QInchworm","category":"page"},{"location":"#QInchworm.jl","page":"Home","title":"QInchworm.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [QInchworm.QInchworm]","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [QInchworm.QInchworm]","category":"page"},{"location":"#QInchworm.QInchworm","page":"Home","title":"QInchworm.QInchworm","text":"A quasi Monte Carlo inchworm impurity solver for multi-orbital fermionic models.\n\n\n\n\n\n","category":"module"},{"location":"#QInchworm.SectorBlockMatrix","page":"Home","title":"QInchworm.SectorBlockMatrix","text":"Complex block matrix stored as a dictionary of non-vanishing blocks.\n\nAn element of the dictionary has the form right block index => (left block index, block).\n\n\n\n\n\n","category":"type"},{"location":"#Base.zeros-Tuple{Type{Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}}, KeldyshED.EDCore}","page":"Home","title":"Base.zeros","text":"zeros(\n    _::Type{Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}},\n    ed::KeldyshED.EDCore\n) -> Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}\n\n\nConstruct a block-diagonal complex matrix, whose block structure is consistent with the invariant subspace partition of a given KeldyshED.EDCore object. All matrix elements of the stored blocks are set to zero.\n\n\n\n\n\n","category":"method"},{"location":"#QInchworm.operator_to_sector_block_matrix-Tuple{KeldyshED.EDCore, KeldyshED.Operators.OperatorExpr}","page":"Home","title":"QInchworm.operator_to_sector_block_matrix","text":"operator_to_sector_block_matrix(\n    ed::KeldyshED.EDCore,\n    op::KeldyshED.Operators.OperatorExpr\n) -> Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}\n\n\nReturns the SectorBlockMatrix representation of the many-body operator.\n\n\n\n\n\n","category":"method"},{"location":"#Sub-Modules","page":"Home","title":"Sub Modules","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"QInchworm.qmc_integrate\nQInchworm.configuration\nQInchworm.ppgf\nQInchworm.diagrammatics","category":"page"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"CurrentModule = QInchworm.configuration","category":"page"},{"location":"modules/configuration/#QInchworm.configuration","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"","category":"section"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"The configuration module is a framework for representing generic pseudo particle diagrams with a back-bone of pseudo particle propagators and insertions of pseudo particle interactions at fixed times.","category":"page"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"There is also functionality to represent Inch-Worm diagrams with a fixed \"inch\" time where the back bone propagator switches from bold to bare.","category":"page"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"The building blocks common for all kinds of diagrams are contained in the Expansion struct. Given an Expansion a particular diagram with fixed times can be represented by the Configuration struct.","category":"page"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"Given an Expansion and a Configuration the eval function can be used to evaluate the value of the \"configuration\"/diagram.","category":"page"},{"location":"modules/configuration/#Pseudo-particle-interactions","page":"QInchworm.configuration","title":"Pseudo particle interactions","text":"","category":"section"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"In pseudo particle strong coupling expansions the notion \"hybridization insertions\" can be seen as a retarded interaction between the pseudo particles.","category":"page"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"One such pseudo particle interaction is defined by a scalar hybridization function Green's function and an initial and final many-body operator.","category":"page"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"Thus, a pseudo particle Expansion is defined, in part, by the list of all possible pseudo particle interactions in the system, where each pseudo particle interaction is represented using the struct InteractionPair.","category":"page"},{"location":"modules/configuration/#Pseudo-particle-symmetry-sectors","page":"QInchworm.configuration","title":"Pseudo-particle symmetry sectors","text":"","category":"section"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"The pseudo-particle propagator and the many-body operators are represented as block matrices in the local Hilbert space. The blocking is performed by KeldyshED.jl based on the symmetries of the local Hamiltonian.","category":"page"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"To represent many-body operators the SectorBlockMatrix type is used.","category":"page"},{"location":"modules/configuration/#Module-Index","page":"QInchworm.configuration","title":"Module Index","text":"","category":"section"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"Modules = [configuration]","category":"page"},{"location":"modules/configuration/#Module-API-Documentation","page":"QInchworm.configuration","title":"Module API Documentation","text":"","category":"section"},{"location":"modules/configuration/","page":"QInchworm.configuration","title":"QInchworm.configuration","text":"Modules = [configuration]","category":"page"},{"location":"modules/configuration/#QInchworm.configuration.Configuration","page":"QInchworm.configuration","title":"QInchworm.configuration.Configuration","text":"struct Configuration\n\nThe Configuration struct defines a single diagram in a peudo-particle expansion with fixed insertions of pseduo-particle interactions and auxilliary operators.\n\nnodes::Vector{QInchworm.configuration.Node}: List of nodes in time with associated operators\npairs::Vector{QInchworm.configuration.NodePair}: List of pairs of nodes in time associated with a hybridization propagator\nparity::Float64: Parity of the diagram p = (-1)^N, with N number of hybridization line crossings\ndeterminants::Vector{QInchworm.configuration.Determinant}: List of groups of time nodes associated with expansion determinants\npaths::Vector{Vector{Tuple{Int64, Int64}}}: List of precomputed trace paths\nsplit_node_idx::Union{Nothing, Int64}: Position of the node that splits the integration domain into two simplices\nop_node_idx::Union{Nothing, Tuple{Int64, Int64}}: Positions of two operator nodes used to measure correlation functions\npair_node_idxs::Vector{Int64}: Positions of nodes coupled by pair interactions\n\n\n\n\n\n","category":"type"},{"location":"modules/configuration/#QInchworm.configuration.InteractionEnum","page":"QInchworm.configuration","title":"QInchworm.configuration.InteractionEnum","text":"Interaction type classification using @enum\n\nPossible values: pair_flag, determinant_flag, identity_flag, inch_flag, operator_flag\n\nprimitive type InteractionEnum <: Enum{Int32} 32\n\n\n\n\n\n","category":"type"},{"location":"modules/configuration/#QInchworm.configuration.Node","page":"QInchworm.configuration","title":"QInchworm.configuration.Node","text":"struct Node\n\nNode in time with associated operator\n\ntime::Keldysh.BranchPoint: Contour time point\noperator_ref::QInchworm.configuration.OperatorReference: Reference to operator\n\n\n\n\n\n","category":"type"},{"location":"modules/configuration/#QInchworm.configuration.Node-Tuple{Keldysh.BranchPoint}","page":"QInchworm.configuration","title":"QInchworm.configuration.Node","text":"Node(\n    time::Keldysh.BranchPoint\n) -> QInchworm.configuration.Node\n\n\nReturns a node at time time::Time with an associated identity operator.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.NodePair","page":"QInchworm.configuration","title":"QInchworm.configuration.NodePair","text":"struct NodePair\n\nNode with pair of times and an associated interaction index.\n\ntime_f::Keldysh.BranchPoint: Final time\ntime_i::Keldysh.BranchPoint: Initial time\nindex::Int64: Index for interaction\n\n\n\n\n\n","category":"type"},{"location":"modules/configuration/#QInchworm.configuration.OperatorReference","page":"QInchworm.configuration","title":"QInchworm.configuration.OperatorReference","text":"struct OperatorReference\n\nLightweight reference to an Expansion operator.\n\nkind::QInchworm.configuration.InteractionEnum: Interaction type of operator\ninteraction_index::Int64: Index for interaction\noperator_index::Int64: Index to operator\n\n\n\n\n\n","category":"type"},{"location":"modules/configuration/#QInchworm.configuration.InchNode-Tuple{Keldysh.BranchPoint}","page":"QInchworm.configuration","title":"QInchworm.configuration.InchNode","text":"InchNode(\n    time::Keldysh.BranchPoint\n) -> QInchworm.configuration.Node\n\n\nReturns an \"inch\" node at time time::Time with an associated identity operator.\n\nThe Inch node triggers the configuration evaluator to switch from bold to bare pseudo particle propagator.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.Nodes-Tuple{QInchworm.configuration.NodePair}","page":"QInchworm.configuration","title":"QInchworm.configuration.Nodes","text":"Nodes(\n    pair::QInchworm.configuration.NodePair\n) -> Vector{QInchworm.configuration.Node}\n\n\nReturns a list of Node's corresponding to the pair pair::NodePair.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.OperatorNode-Tuple{Keldysh.BranchPoint, Int64, Int64}","page":"QInchworm.configuration","title":"QInchworm.configuration.OperatorNode","text":"OperatorNode(\n    time::Keldysh.BranchPoint,\n    interaction_index::Int64,\n    operator_index::Int64\n) -> QInchworm.configuration.Node\n\n\nReturns an operator node at time time::Time with an associated operator.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.eval-Tuple{QInchworm.expansion.Expansion, QInchworm.configuration.Configuration}","page":"QInchworm.configuration","title":"QInchworm.configuration.eval","text":"eval(\n    exp::QInchworm.expansion.Expansion,\n    conf::QInchworm.configuration.Configuration\n) -> Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}\n\n\nEvaluate the configuration conf in the pseud-particle expansion exp.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.get_block_matrix-Tuple{QInchworm.expansion.Expansion, QInchworm.configuration.Node}","page":"QInchworm.configuration","title":"QInchworm.configuration.get_block_matrix","text":"get_block_matrix(\n    exp::QInchworm.expansion.Expansion,\n    node::QInchworm.configuration.Node\n) -> Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}\n\n\nReturns the SectorBlockMatrix representation of the many-body operator at the given `node::Node'.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.is_inch_node-Tuple{QInchworm.configuration.Node}","page":"QInchworm.configuration","title":"QInchworm.configuration.is_inch_node","text":"is_inch_node(node::QInchworm.configuration.Node) -> Bool\n\n\nReturns true if the node is an \"inch\" node.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.is_operator_node-Tuple{QInchworm.configuration.Node}","page":"QInchworm.configuration","title":"QInchworm.configuration.is_operator_node","text":"is_operator_node(node::QInchworm.configuration.Node) -> Bool\n\n\nReturns true if the node is an operator node.\n\n\n\n\n\n","category":"method"},{"location":"modules/configuration/#QInchworm.configuration.sector_block_matrix_from_ppgf-Union{Tuple{MatrixGF}, Tuple{Keldysh.BranchPoint, Keldysh.BranchPoint, Vector{MatrixGF}}} where MatrixGF<:Keldysh.AbstractTimeGF{ComplexF64, false}","page":"QInchworm.configuration","title":"QInchworm.configuration.sector_block_matrix_from_ppgf","text":"sector_block_matrix_from_ppgf(\n    z2::Keldysh.BranchPoint,\n    z1::Keldysh.BranchPoint,\n    P::Array{MatrixGF<:Keldysh.AbstractTimeGF{ComplexF64, false}, 1}\n) -> Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}\n\n\nReturns the SectorBlockMatrix representation of the pseudo particle propagator evaluated at the times z1 and z2.\n\n\n\n\n\n","category":"method"}]
}