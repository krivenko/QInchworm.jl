
module configuration

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf

# -- Exports

#export Expansion
#export InteractionEnum
#export InteractionPair, InteractionPairs
#export InteractionDeterminant, InteractionDeterminants

#export Configuration
#export InchNode
#export Node, Nodes
#export NodePair, NodePairs
#export Determinant, Determinants

# -- Types

const Time = kd.BranchPoint
const Times = Vector{Time}

const ScalarGF = kd.FullTimeGF{ComplexF64, true}
const MatrixGF = kd.GenericTimeGF{ComplexF64, false}
const SectorGF = Vector{MatrixGF}

const Operator = op.RealOperatorExpr
const Operators = Vector{Operator}

const OperatorBlocks = Dict{Tuple{Int64, Int64}, Matrix{Float64}}
const SectorBlockMatrix = Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}

@enum InteractionEnum pair_flag=1 determinant_flag=2 identity_flag=3 inch_flag=4

struct InteractionPair
  operator_f::Operator
  operator_i::Operator
  propagator::ScalarGF
end

function Base.getindex(pair::InteractionPair, idx::Int64)
    return [pair.operator_i, pair.operator_f][idx]
end

const InteractionPairs = Vector{InteractionPair}

struct InteractionDeterminant
  operators_f::Operators
  operators_i::Operators
  propagator::MatrixGF
end

const InteractionDeterminants = Vector{InteractionDeterminant}

struct Expansion
  ed::ked.EDCore
  P0::SectorGF
  P::SectorGF
  pairs::InteractionPairs
  determinants::InteractionDeterminants
  function Expansion(ed::ked.EDCore, grid::kd.FullTimeGrid, interaction_pairs::InteractionPairs)
    P0 = ppgf.atomic_ppgf(grid, ed)
    P = deepcopy(P0)
    return new(ed, P0, P, interaction_pairs, [])
  end
end

struct OperatorReference
  kind::InteractionEnum
  interaction_index::Int64
  operator_index::Int64
end

struct Node
  time::Time
  operator_ref::OperatorReference
end

function Node(time::Time)
    return Node(time, OperatorReference(identity_flag, -1, -1))
end

function InchNode(time::Time)
    return Node(time, OperatorReference(inch_flag, -1, -1))
end

function is_inch_node(node::Node)
    return node.operator_ref.kind == inch_flag
end

const Nodes = Vector{Node}

struct NodePair
  time_f::Time
  time_i::Time
  index::Int64
end

const NodePairs = Vector{NodePair}

function Nodes(pair::NodePair)
    n1 = Node(pair.time_i, OperatorReference(pair_flag, pair.index, 1))
    n2 = Node(pair.time_f, OperatorReference(pair_flag, pair.index, 2))
    return Nodes([n1, n2])
end

struct Determinant
  times_f::Times
  times_i::Times
  interaction_index::Int64
  matrix::Array{ComplexF64, 2}
end

const Determinants = Vector{Determinant}

struct Configuration
    nodes::Nodes
    pairs::NodePairs
    determinants::Determinants
    function Configuration(single_nodes::Nodes, pairs::NodePairs)
        nodes::Nodes = deepcopy(single_nodes)

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

        sort!(nodes, by = n -> twisted_contour_relative_order(n.time))
        return new(nodes, pairs, [])
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

function eval(exp::Expansion, pairs::NodePairs)
    val::ComplexF64 = 1.
    for pair in pairs
        #val *= exp.pairs[pair.index].propagator(pair.time_f, pair.time_i)
        sign = (-1.)^(pair.time_f < pair.time_i)
        val *= im * sign * exp.pairs[pair.index].propagator(pair.time_f, pair.time_i)
    end
    return val
end

function operator_to_sector_block_matrix(exp::Expansion, op::Operator)

    sbm = SectorBlockMatrix()

    op_blocks = ked.operator_blocks(exp.ed, op)

    for ((s_f, s_i), mat) in op_blocks
        sbm[s_i] = (s_f, mat)
    end

    return sbm
end

function operator(exp::Expansion, node::Node)
    op::Operator = Operator()
    if node.operator_ref.kind == pair_flag
        op = exp.pairs[node.operator_ref.interaction_index][node.operator_ref.operator_index]
    elseif node.operator_ref.kind == determinant_flag
        op = exp.determinants[node.operator_ref.interaction_index][node.operator_ref.operator_index]
    elseif node.operator_ref.kind == identity_flag || is_inch_node(node)
        op = Operator(1.)
    else
        throw(BoundsError())
    end
    return operator_to_sector_block_matrix(exp, op)
end

function Base.:+(A::SectorBlockMatrix, B::SectorBlockMatrix)
    return merge((a,b) -> (a[1], a[2] + b[2]), A, B)
end

function Base.:*(A::SectorBlockMatrix, B::SectorBlockMatrix)
    C = SectorBlockMatrix()
    for (s_i, (s, B_mat)) in B
        if haskey(A, s)
            s_f, A_mat = A[s]
            C[s_i] = (s_f, A_mat * B_mat)
        end
    end
    return C
end

function Base.:*(A::Number, B::SectorBlockMatrix)
    C = SectorBlockMatrix()
    for (s_i, (s_f, B_mat)) in B
        C[s_i] = (s_f, A * B_mat)
    end
    return C
end

function Base.:*(A::SectorBlockMatrix, B::Number)
    return B * A
end

function sector_block_matrix_from_ppgf(z2::Time, z1::Time, P::SectorGF)
    M = SectorBlockMatrix()
    for (sidx, p) in enumerate(P)
        M[sidx] = (sidx, p(z2, z1))
    end
    return M
end

function eval(exp::Expansion, nodes::Nodes)

    node = first(nodes)
    val = operator(exp, node)

    P = exp.P # Inch with dressed ppsc propagator from start

    prev_node = node

    for (nidx, node) in enumerate(nodes[2:end])

        if is_inch_node(prev_node)
            P = exp.P0 # Inch with bare ppsc propagator after inch time node
        end

        op = operator(exp, node)
        P_interp = sector_block_matrix_from_ppgf(node.time, prev_node.time, P)

        val = (im * op * P_interp) * val

        prev_node = node
    end

    return -im * val
end

function eval(exp::Expansion, conf::Configuration)
    return eval(exp, conf.pairs) * eval(exp, conf.nodes)
end

end # module configuration
