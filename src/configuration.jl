module configuration

using DocStringExtensions

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm: SectorBlockMatrix
using QInchworm.ppgf
using QInchworm.expansion: Operator, Expansion
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
    return [idx for (idx, node) in enumerate(nodes)
            if node.operator_ref.kind == pair_flag] |> reverse
end

const Path = Vector{Tuple{Int, Int}}

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

    "List of precomputed trace paths"
    paths::Vector{Path}

    "Position of the node that splits the integration domain into two simplices"
    split_node_idx::Union{Int, Nothing}
    "Positions of two operator nodes used to measure correlation functions"
    op_node_idx::Union{Tuple{Int, Int}, Nothing}
    "Positions of nodes coupled by pair interactions"
    pair_node_idxs::Vector{Int}

    """
    $(TYPEDSIGNATURES)

    Construct a configuration for a bold PPGF calculation.

    Parameters
    ----------

    diagram :  Diagram to derive the configuration from.
    exp :      Pseudo-particle expansion problem.
    d_before : Number of pair nodes before the inchworm node. When omitted, no inchworm node
               will be present in the configuration.
    """
    function Configuration(diagram::Diagram,
                           exp::Expansion,
                           d_before::Union{Int, Nothing} = nothing)
        # All nodes are initially placed at the start of the contour
        t_0 = first(exp.P).grid.contour(0.0)

        n_i, n_w, n_f = Node(t_0), InchNode(t_0), Node(t_0)

        inch_node_idx = isnothing(d_before) ? nothing : (d_before + 2)

        pairs = [ NodePair(t_0, t_0, diagram.pair_idxs[idx])
                  for (idx, (a, b)) in enumerate(diagram.topology.pairs) ]
        parity = diagram.topology.parity

        n = 2 * diagram.topology.order
        pairnodes = [ Node(t_0) for i in 1:n ]

        for (idx, (f_idx, i_idx)) in enumerate(diagram.topology.pairs)
            p_idx = diagram.pair_idxs[idx]
            pairnodes[i_idx] = Node(t_0, OperatorReference(pair_flag, p_idx, 1))
            pairnodes[f_idx] = Node(t_0, OperatorReference(pair_flag, p_idx, 2))
        end

        reverse!(pairnodes)

        if length(pairnodes) > 0
            if !isnothing(inch_node_idx)
                nodes = vcat([n_i],
                             pairnodes[1:d_before],
                             [n_w],
                             pairnodes[d_before+1:end],
                             [n_f])
            else
                nodes = vcat([n_i], pairnodes, [n_f])
            end
        else
            nodes = !isnothing(inch_node_idx) ? [n_i, n_w, n_f] : [n_i, n_f]
        end

        paths = get_paths(exp, nodes)
        pair_node_idxs = get_pair_node_idxs(nodes)

        return new(nodes, pairs, parity, [], paths, inch_node_idx, nothing, pair_node_idxs)
    end

    """
    $(TYPEDSIGNATURES)

    Construct a configuration for a GF calculation.

    Parameters
    ----------

    diagram :       Diagram to derive the configuration from.
    exp :           Pseudo-particle expansion problem.
    d_before :      Number of pair nodes before the split node.
    op_pair_index : Index of the C / C^+ operator pair within exp.corr_operators.
    """
    function Configuration(diagram::Diagram,
                           exp::Expansion,
                           d_before::Integer,
                           op_pair_index::Integer)
        # All nodes are initially placed at the start of the contour
        t_0 = first(exp.P).grid.contour(0.0)

        n_cdag = OperatorNode(t_0, op_pair_index, 2)
        n_c = OperatorNode(t_0, op_pair_index, 1)
        n_f = Node(t_0)

        pairs = [NodePair(t_0, t_0, diagram.pair_idxs[idx])
                 for (idx, (a, b)) in enumerate(diagram.topology.pairs)]
        parity = diagram.topology.parity

        n = 2 * diagram.topology.order
        pairnodes = [Node(t_0) for i in 1:n]

        for (idx, (f_idx, i_idx)) in enumerate(diagram.topology.pairs)
            p_idx = diagram.pair_idxs[idx]
            pairnodes[i_idx] = Node(t_0, OperatorReference(pair_flag, p_idx, 1))
            pairnodes[f_idx] = Node(t_0, OperatorReference(pair_flag, p_idx, 2))
        end

        reverse!(pairnodes)

        if length(pairnodes) > 0
            nodes = vcat([n_cdag],
                         pairnodes[1:d_before],
                         [n_c],
                         pairnodes[d_before+1:end],
                         [n_f])
        else
            nodes = [n_cdag, n_c, n_f]
        end

        split_node_idx = d_before + 2

        paths = get_paths(exp, nodes)
        pair_node_idxs = get_pair_node_idxs(nodes)

        return new(nodes,
                   pairs,
                   parity,
                   [],
                   paths,
                   split_node_idx,
                   (1, d_before + 2),
                   pair_node_idxs)
    end
end

function set_initial_node_time!(configuration::Configuration, t_i::Time)
    @assert configuration.nodes[1].operator_ref.kind == identity_flag
    configuration.nodes[1] = Node(t_i)
end

function set_final_node_time!(configuration::Configuration, t_f::Time)
    @assert configuration.nodes[end].operator_ref.kind == identity_flag
    configuration.nodes[end] = Node(t_f)
end

function set_inchworm_node_time!(configuration::Configuration, t_w::Time)
    @assert !isnothing(configuration.split_node_idx)
    @assert is_inch_node(configuration.nodes[configuration.split_node_idx])
    configuration.nodes[configuration.split_node_idx] = InchNode(t_w)
end

function set_operator_node_time!(configuration::Configuration, idx::Integer, t::Time)
    @boundscheck 1 <= idx <= 2
    @assert !isnothing(configuration.op_node_idx)
    op_ref = configuration.nodes[configuration.op_node_idx[idx]].operator_ref
    configuration.nodes[configuration.op_node_idx[idx]] = Node(t, op_ref)
end

function update_pair_node_times!(configuration::Configuration, diagram::Diagram, times::Vector{Time})
    for (t_idx, n_idx) in enumerate(configuration.pair_node_idxs)
        op_ref = configuration.nodes[n_idx].operator_ref
        configuration.nodes[n_idx] = Node(times[t_idx], op_ref)
    end

    for (p_idx, (idx_tf, idx_ti)) in enumerate(diagram.topology.pairs)
        int_idx = configuration.pairs[p_idx].index
        configuration.pairs[p_idx] = NodePair(times[idx_tf], times[idx_ti], int_idx)
    end
end

function eval(exp::Expansion, pairs::Vector{NodePair}, parity::Float64)

    order = length(pairs)
    val::ComplexF64 = parity * (-1)^order

    for pair in pairs
        val *= im * exp.pairs[pair.index].propagator(pair.time_f, pair.time_i)
    end
    return val
end

"""
$(TYPEDSIGNATURES)

Returns the [`SectorBlockMatrix`](@ref) representation of the many-body operator at
the given `node::Node'.
"""
function get_block_matrix(exp::Expansion, node::Node)::SectorBlockMatrix
    op_ref = node.operator_ref
    if op_ref.kind == pair_flag
        return exp.pair_operator_mat[op_ref.interaction_index][op_ref.operator_index]
    elseif op_ref.kind == operator_flag
        return exp.corr_operators_mat[op_ref.interaction_index][op_ref.operator_index]
    elseif op_ref.kind == identity_flag || is_inch_node(node)
        return exp.identity_mat
    else
        throw(BoundsError())
    end
end

"""
$(TYPEDSIGNATURES)

Returns the [`SectorBlockMatrix`](@ref) representation of the pseudo particle propagator
evaluated at the times `z1` and `z2`.
"""
function sector_block_matrix_from_ppgf(z2::Time, z1::Time,
                                       P::Vector{MatrixGF})::SectorBlockMatrix where {
    MatrixGF <: kd.AbstractTimeGF{ComplexF64, false}}
    M = SectorBlockMatrix()
    for (sidx, p) in enumerate(P)
        M[sidx] = (sidx, p(z2, z1))
    end
    return M
end

function eval(exp::Expansion, nodes::Vector{Node})

    node = first(nodes)
    val = get_block_matrix(exp, node)

    P = exp.P # Inch with dressed ppsc propagator from start

    prev_node = node

    for (nidx, node) in enumerate(nodes[2:end])

        op = get_block_matrix(exp, node)
        P_interp = sector_block_matrix_from_ppgf(node.time, prev_node.time, P)

        val = (im * op * P_interp) * val

        prev_node = node
    end

    return -im * val
end

function get_paths(exp::Expansion, nodes::Vector{Node})::Vector{Path}

    operators = [get_block_matrix(exp, node) for node in nodes]
    N_sectors = length(exp.P)

    paths = Path[]
    for s_i in 1:N_sectors
        path = Path()
        for operator in operators
            if haskey(operator, s_i)
                s_f, op_mat = operator[s_i]
                push!(path, (s_i, s_f))
                s_i = s_f
            end
        end
        if length(path) == length(operators)
            push!(paths, path)
        end
    end
    return paths
end

function eval(exp::Expansion,
              nodes::Vector{Node},
              paths::Vector{Path},
              has_split_node::Bool)::SectorBlockMatrix

    val = SectorBlockMatrix()

    for path in paths
        prev_node = first(nodes)
        first_s_i, s_f = first(path)
        mat = get_block_matrix(exp, first(nodes))[first_s_i][2]

        for (nidx, node) in enumerate(nodes[2:end])

            s_i, s_f = path[nidx + 1]
            op_mat = get_block_matrix(exp, node)[s_i][2]

            P_interp = has_split_node ? exp.P[s_i](node.time, prev_node.time) :
                                        exp.P0[s_i](node.time, prev_node.time)

            mat = im * op_mat * P_interp * mat

            prev_node = node
        end
        val[first_s_i] = (s_f, -im * mat)
    end

    return val
end

function eval_acc!(val::SectorBlockMatrix, scalar::ComplexF64,
                   exp::Expansion, nodes::Vector{Node},
                   paths::Vector{Path},
                   has_split_node::Bool)

    P = has_split_node ? exp.P : exp.P0

    @inbounds for path in paths
        prev_node = first(nodes)
        S_i, S_f = first(path)
        mat = get_block_matrix(exp, first(nodes))[S_i][2]

        for (nidx, node) in enumerate(nodes[2:end])

            s_i, s_f = path[nidx + 1]
            op_mat = get_block_matrix(exp, node)[s_i][2]

            if node.time.ref >= prev_node.time.ref
                P_interp = P[s_i](node.time, prev_node.time)
            else
                # Fix for equal time nodes where order is flipped
                # from float roundoff
                P_interp = P[s_i](prev_node.time, node.time)
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
           eval(exp, conf.nodes, conf.paths, !isnothing(conf.split_node_idx))
end

function eval_acc!(value::SectorBlockMatrix, exp::Expansion, conf::Configuration)
    scalar::ComplexF64 = eval(exp, conf.pairs, conf.parity)
    eval_acc!(value, scalar, exp, conf.nodes, conf.paths, !isnothing(conf.split_node_idx))
end

function eval(expansion::Expansion,
    diagrams::Vector{Diagram},
    configurations::Vector{Configuration},
    times::Vector{Time},
)::SectorBlockMatrix
    value = zeros(SectorBlockMatrix, expansion.ed)

    for (diagram, configuration) in zip(diagrams, configurations)
        update_pair_node_times!(configuration, diagram, times)
        eval_acc!(value, expansion, configuration)
    end

    return value
end

end # module configuration
