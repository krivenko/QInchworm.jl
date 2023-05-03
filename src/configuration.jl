module configuration

using DocStringExtensions

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm: SectorBlockMatrix
using QInchworm.ppgf
using QInchworm.expansion: Operator, Expansion, operator_to_sector_block_matrix
using QInchworm.diagrammatics: Diagram, n_crossings

const Time = kd.BranchPoint

"""
Interaction type classification using `@enum`

Possible values: `pair_flag`, `determinant_flag`, `identity_flag`, `inch_flag`, `operator_flag`

$(TYPEDEF)
"""
@enum InteractionEnum pair_flag=1 determinant_flag=2 identity_flag=3 inch_flag=4 operator_flag=5

"""
$(TYPEDEF)

Lightweight reference to an `Expansion` operator.

$(TYPEDFIELDS)
"""
struct OperatorReference
    "Interaction type of operator"
    kind::InteractionEnum
    "Index for interaction"
    interaction_index::Int64
    "Index to operator"
    operator_index::Int64
end

"""
$(TYPEDEF)

Node in time with associated operator

$(TYPEDFIELDS)
"""
struct Node
    "Contour time point"
    time::Time
    "Reference to operator"
    operator_ref::OperatorReference
end

"""
$(TYPEDSIGNATURES)

Returns a node at time `time::Time` with an associated identity operator.
"""
function Node(time::Time)::Node
    return Node(time, OperatorReference(identity_flag, -1, -1))
end

"""
$(TYPEDSIGNATURES)

Returns an operator node at time `time::Time` with an associated operator.
"""
function OperatorNode(time::Time,
                      interaction_index::Int64,
                      operator_index::Int64)::Node
    return Node(time, OperatorReference(operator_flag, interaction_index, operator_index))
end

"""
$(TYPEDSIGNATURES)

Returns an "inch" node at time `time::Time` with an associated identity operator.

The Inch node triggers the configuration evaluator to switch from bold to bare pseudo particle propagator.
"""
function InchNode(time::Time)::Node
    return Node(time, OperatorReference(inch_flag, -1, -1))
end

"""
$(TYPEDSIGNATURES)

Returns `true` if the node is an "inch" node.
"""
function is_inch_node(node::Node)::Bool
    return node.operator_ref.kind == inch_flag
end

"""
$(TYPEDSIGNATURES)

Returns `true` if the node is an operator node.
"""
function is_operator_node(node::Node)::Bool
    return node.operator_ref.kind == operator_flag
end

"""
$(TYPEDEF)

Node with pair of times and an associated interaction index.

$(TYPEDFIELDS)
"""
struct NodePair
    "Final time"
    time_f::Time
    "Initial time"
    time_i::Time
    "Index for interaction"
    index::Int64
end

"""
$(TYPEDSIGNATURES)

Returns a list of `Node`'s corresponding to the pair `pair::NodePair`.
"""
function Nodes(pair::NodePair)
    n1 = Node(pair.time_i, OperatorReference(pair_flag, pair.index, 1))
    n2 = Node(pair.time_f, OperatorReference(pair_flag, pair.index, 2))
    return [n1, n2]
end

struct Determinant
    times_f::Vector{Time}
    times_i::Vector{Time}
    interaction_index::Int64
    matrix::Matrix{ComplexF64}
end

function get_pair_node_idxs(nodes::Vector{Node})::Vector{Int}
    reverse([idx for (idx, node) in enumerate(nodes) if node.operator_ref.kind == pair_flag])
end

const Path = Vector{Tuple{Int, Int, Matrix{ComplexF64}}}

"""
$(TYPEDEF)

The `Configuration` struct defines a single diagram in a peudo-particle
expansion with fixed insertions of pseduo-particle interactions and
auxilliary operators.

$(TYPEDFIELDS)
"""
struct Configuration
    "List of nodes in time with associated operators"
    nodes::Vector{Node}
    "List of pairs of nodes in time associated with a hybridization propagator"
    pairs::Vector{NodePair}
    "Parity of the diagram p = (-1)^N, with N number of hybridization line crossings"
    parity::Float64
    "List of groups of time nodes associated with expansion determinants"
    determinants::Vector{Determinant}

    #"List of node operators"
    #operators::Vector{SectorBlockMatrix}

    "List of precomputed trace paths with operator sub matrices"
    paths::Vector{Path}

    "Position of the node that splits the integration domain into two simplices"
    split_node_idx::Union{Int, Nothing}

    op_node_idx::Union{Tuple{Int, Int}, Nothing}

    node_idxs::Vector{Int}

    # TODO: Refactor and document these constructors. Also, do we really need 3 of these?
    function Configuration(single_nodes::Vector{Node}, pairs::Vector{NodePair}, exp::Expansion)
        nodes = deepcopy(single_nodes)

        for pair in pairs
            append!(nodes, Nodes(pair))
        end

        function twisted_contour_relative_order(time::kd.BranchPoint)
            if time.domain == kd.backward_branch
                return 1. - time.ref
            elseif time.domain == kd.forward_branch
                return 2. + time.ref
            elseif time.domain == kd.imaginary_branch
                return 1. + time.ref
            else
                throw(BoundsError())
            end
        end

        sort!(nodes, by = n -> twisted_contour_relative_order(n.time), alg=MergeSort)

        has_inch_node = any(is_inch_node.(nodes))
        has_operator_node = any(is_operator_node.(nodes))

        split_node_idx = (has_inch_node || has_operator_node) ? 3 : nothing

        paths = get_paths(exp, nodes)
        n_idxs = get_pair_node_idxs(nodes)

        parity = 1.0
        return new(nodes, pairs, parity, [], paths, split_node_idx, nothing, n_idxs)
    end
    function Configuration(diagram::Diagram, exp::Expansion, d_before::Int)

        contour = first(exp.P).grid.contour
        time = contour(0.0)
        n_f, n_w, n_i = Node(time), InchNode(time), Node(time)

        split_node_idx = (d_before > 0) ? (d_before + 2) : nothing
        single_nodes = (split_node_idx !== nothing) ? [n_i, n_w, n_f] : [n_i, n_f]

        pairs = [ NodePair(time, time, diagram.pair_idxs[idx])
                  for (idx, (a, b)) in enumerate(diagram.topology.pairs) ]
        parity = diagram.topology.parity

        n = diagram.topology.order*2
        pairnodes = [ Node(time) for i in 1:n ]

        for (idx, (f_idx, i_idx)) in enumerate(diagram.topology.pairs)
            p_idx = diagram.pair_idxs[idx]
            pairnodes[i_idx] = Node(time, OperatorReference(pair_flag, p_idx, 1))
            pairnodes[f_idx] = Node(time, OperatorReference(pair_flag, p_idx, 2))
        end

        reverse!(pairnodes)

        if length(pairnodes) > 0
            if split_node_idx !== nothing
                nodes = vcat([n_i], pairnodes[1:d_before], [n_w], pairnodes[d_before+1:end], [n_f])
            else
                nodes = vcat([n_i], pairnodes, [n_f])
            end
        else
            nodes = single_nodes
        end

        paths = get_paths(exp, nodes)
        n_idxs = get_pair_node_idxs(nodes)

        return new(nodes, pairs, parity, [], paths, split_node_idx, nothing, n_idxs)
    end
    function Configuration(diagram::Diagram, exp::Expansion, d_before::Int, op_pair_index::Int)
        contour = first(exp.P).grid.contour
        time = contour(0.0)

        n_f = Node(time)
        n_c = OperatorNode(time, op_pair_index, 1)
        n_cdag = OperatorNode(time, op_pair_index, 2)

        single_nodes = [n_cdag, n_c, n_f]

        pairs = [ NodePair(time, time, diagram.pair_idxs[idx])
        for (idx, (a, b)) in enumerate(diagram.topology.pairs) ]
        parity = diagram.topology.parity

        n = diagram.topology.order*2
        pairnodes = [ Node(time) for i in 1:n ]

        for (idx, (f_idx, i_idx)) in enumerate(diagram.topology.pairs)
            p_idx = diagram.pair_idxs[idx]
            pairnodes[i_idx] = Node(time, OperatorReference(pair_flag, p_idx, 1))
            pairnodes[f_idx] = Node(time, OperatorReference(pair_flag, p_idx, 2))
        end

        if length(pairnodes) > 0
            nodes = vcat([n_cdag], pairnodes[1:d_before], [n_c], pairnodes[d_before+1:end], [n_f])
        else
            nodes = single_nodes
        end

        split_node_idx = d_before + 2

        paths = get_paths(exp, nodes)
        n_idxs = get_pair_node_idxs(nodes)

        return new(nodes, pairs, parity, [], paths, split_node_idx, (1, d_before + 2), n_idxs)
    end
end

# TODO: Can we use kd.heaviside() instead?
function Base.isless(t1::kd.BranchPoint, t2::kd.BranchPoint)
    if t1.domain > t2.domain
        return true
    elseif t2.domain < t2.domain
        return false
    else # same domain
        if t1.domain == kd.forward_branch
            return real(t1.val) < real(t2.val)
        elseif t1.domain == kd.backward_branch
            return real(t1.val) > real(t2.val)
        else
            return imag(t1.val) > imag(t2.val)
        end
    end
end

function eval(exp::Expansion, pairs::Vector{NodePair}, parity::Float64)

    order = length(pairs)
    val::ComplexF64 = parity * (-1)^order

    for pair in pairs
        #@assert pair.time_f >= pair.time_i
        val *= im * exp.pairs[pair.index].propagator(pair.time_f, pair.time_i)
    end
    return val
end

"""
$(TYPEDSIGNATURES)

Returns the [`SectorBlockMatrix`](@ref) representation of the many-body operator at the given time `node::Node'.
"""
function operator(exp::Expansion, node::Node)::SectorBlockMatrix
    op::Operator = Operator()
    if node.operator_ref.kind == pair_flag
        op = exp.pairs[node.operator_ref.interaction_index][node.operator_ref.operator_index]
    elseif node.operator_ref.kind == determinant_flag
        op = exp.determinants[node.operator_ref.interaction_index][node.operator_ref.operator_index]
    elseif node.operator_ref.kind == operator_flag
        op = exp.corr_operators[node.operator_ref.interaction_index][node.operator_ref.operator_index]
    elseif node.operator_ref.kind == identity_flag || is_inch_node(node)
        op = Operator(1.)
    else
        throw(BoundsError())
    end
    return operator_to_sector_block_matrix(exp, op)
end

"""
$(TYPEDSIGNATURES)

Returns the [`SectorBlockMatrix`](@ref) representation of the pseudo particle propagator evaluated at the times `z1` and `z2`.
"""
function sector_block_matrix_from_ppgf(z2::Time, z1::Time, P::Vector{MatrixGF}) where {
    MatrixGF <: kd.AbstractTimeGF{ComplexF64, false}}
    M = SectorBlockMatrix()
    for (sidx, p) in enumerate(P)
        M[sidx] = (sidx, p(z2, z1))
    end
    return M
end

function eval(exp::Expansion, nodes::Vector{Node})

    node = first(nodes)
    val = operator(exp, node)

    P = exp.P # Inch with dressed ppsc propagator from start

    prev_node = node

    for (nidx, node) in enumerate(nodes[2:end])

        op = operator(exp, node)
        P_interp = sector_block_matrix_from_ppgf(node.time, prev_node.time, P)

        val = (im * op * P_interp) * val

        prev_node = node
    end

    return -im * val
end

function get_paths(exp::Expansion, nodes::Vector{Node})::Vector{Path}

    operators = [ operator(exp, node) for node in nodes ]
    N_sectors = length(exp.P)

    paths = Path[]
    for s_i in 1:N_sectors
        path = Path()
        for operator in operators
            if haskey(operator, s_i)
                s_f, op_mat = operator[s_i]
                push!(path, (s_i, s_f, op_mat))
                s_i = s_f
            end
        end
        if length(path) == length(operators)
            push!(paths, path)
        end
    end
    return paths
end

function eval(exp::Expansion, nodes::Vector{Node}, paths::Vector{Vector{Tuple{Int, Int, Matrix{ComplexF64}}}}, has_split_node::Bool)

    start = operator(exp, first(nodes))
    val = SectorBlockMatrix()

    for path in paths

        prev_node = first(nodes)
        first_s_i, s_f, mat = first(path)

        for (nidx, node) in enumerate(nodes[2:end])

            s_i, s_f, op_mat = path[nidx + 1]

            #@assert node.time >= prev_node.time

            P_interp = has_split_node ? exp.P[s_i](node.time, prev_node.time) : exp.P0[s_i](node.time, prev_node.time)

            mat = im * op_mat * P_interp * mat

            prev_node = node
        end
        val[first_s_i] = (s_f, -im * mat)
    end

    return val
end

function eval_acc!(val::SectorBlockMatrix, scalar::ComplexF64,
                   exp::Expansion, nodes::Vector{Node},
                   paths::Vector{Vector{Tuple{Int, Int, Matrix{ComplexF64}}}},
                   has_split_node::Bool)

    start = operator(exp, first(nodes))

    P = has_split_node ? exp.P : exp.P0

    @inbounds for path in paths

        prev_node = first(nodes)
        S_i, S_f, mat = first(path)

        for (nidx, node) in enumerate(nodes[2:end])

            s_i, s_f, op_mat = path[nidx + 1]

            #@assert !(node.time < prev_node.time)
            #@assert node.time >= prev_node.time # BROKEN FIXME ?

            if node.time.ref >= prev_node.time.ref
                P_interp = P[s_i](node.time, prev_node.time)
            else
                # Fix for equal time nodes where order is flipped
                # from float roundoff
                #@assert (node.time.ref - prev_node.time.ref) > -1e-10
                P_interp = P[s_i](prev_node.time, node.time) # reversed node order
            end

            mat = im * op_mat * P_interp * mat

            prev_node = node
        end
        val[S_i][2] .+= -im * scalar * mat
    end

    return val
end


"""
$(TYPEDSIGNATURES)

Evaluate the configuration `conf` in the pseud-particle expansion `exp`.
"""
function eval(exp::Expansion, conf::Configuration)::SectorBlockMatrix
    return eval(exp, conf.pairs, conf.parity) *
           eval(exp, conf.nodes, conf.paths, conf.split_node_idx !== nothing)
end


function eval_acc!(value::SectorBlockMatrix, exp::Expansion, conf::Configuration)
    scalar::ComplexF64 = eval(exp, conf.pairs, conf.parity)
    eval_acc!(value, scalar, exp, conf.nodes, conf.paths, conf.split_node_idx !== nothing)
    return
end

end # module configuration
