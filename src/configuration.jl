module configuration

using DocStringExtensions
using LinearAlgebra: norm

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf
import QInchworm.spline_gf: SplineInterpolatedGF
import QInchworm.spline_gf: IncSplineImaginaryTimeGF, extend!

import QInchworm.diagrammatics: Diagram, Diagrams
import QInchworm.diagrammatics; diag = diagrammatics

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

const Operator = op.RealOperatorExpr
const Operators = Vector{Operator}

const OperatorBlocks = Dict{Tuple{Int64, Int64}, Matrix{Float64}}

""" Representation of local many-body operator in terms of block matrices.

See also: [`operator_to_sector_block_matrix`](@ref) """
const SectorBlockMatrix = Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}

const AllPPGFSectorTypes = Union{
    ppgf.FullTimePPGFSector,
    ppgf.ImaginaryTimePPGFSector,
    SplineInterpolatedGF{ppgf.FullTimePPGFSector, ComplexF64, false},
    SplineInterpolatedGF{ppgf.ImaginaryTimePPGFSector, ComplexF64, false},
    IncSplineImaginaryTimeGF{ComplexF64, false}
}

const AllPPGFTypes = Union{
    Vector{ppgf.FullTimePPGFSector},
    Vector{ppgf.ImaginaryTimePPGFSector},
    Vector{SplineInterpolatedGF{ppgf.FullTimePPGFSector, ComplexF64, false}},
    Vector{SplineInterpolatedGF{ppgf.ImaginaryTimePPGFSector, ComplexF64, false}},
    Vector{IncSplineImaginaryTimeGF{ComplexF64, false}}
}

function Base.zero(P::T) where T <: AllPPGFTypes
    [ zero(p) for p in P ]
end

"""
Interaction type classification using `@enum`

Possible values: `pair_flag`, `determinant_flag`, `identity_flag`, `inch_flag`

$(TYPEDEF)
"""
@enum InteractionEnum pair_flag=1 determinant_flag=2 identity_flag=3 inch_flag=4

"""
$(TYPEDEF)

Data type for pseudo-particle interactions, containing two operators and one scalar propagator.

$(TYPEDFIELDS)
"""
struct InteractionPair{ScalarGF <: kd.AbstractTimeGF{ComplexF64, true}}
  "Final time operator"
  operator_f::Operator
  "Initial time operator"
  operator_i::Operator
  "Scalar propagator"
  propagator::ScalarGF
end

function Base.getindex(pair::InteractionPair, idx::Int64)
    return [pair.operator_i, pair.operator_f][idx]
end

"""
$(TYPEDEF)
"""
const InteractionPairs = Vector{InteractionPair{ScalarGF}} where ScalarGF

struct InteractionDeterminant{PPGFSector <: AllPPGFSectorTypes}
  operators_f::Operators
  operators_i::Operators
  propagator::PPGFSector
end

const InteractionDeterminants = Vector{InteractionDeterminant}

"""
$(TYPEDEF)

The `Expansion` struct contains the components needed to define
a pseudo-particle expansion problem.

$(TYPEDFIELDS)
"""
struct Expansion{ScalarGF <: kd.AbstractTimeGF{ComplexF64, true}, PPGF <: AllPPGFTypes}
  "Exact diagonalization solver for the local degrees of freedom"
  ed::ked.EDCore
  "Non-interacting pseudo-particle Green's function"
  P0::PPGF
  "Interacting pseudo-particle Green's function"
  P::PPGF
  "Contribution to interacting pseudo-particle Green's function, per diagram order"
  P_orders::Vector{PPGF}
  "List of pseudo-particle interactions"
  pairs::InteractionPairs{ScalarGF}
  "List of hybridization function determinants (not implemented yet)"
  determinants::Vector{InteractionDeterminant}
  """
  $(TYPEDSIGNATURES)
  """
  function Expansion(ed::ked.EDCore,
                     grid::kd.AbstractTimeGrid,
                     interaction_pairs::InteractionPairs{ScalarGF};
                     interpolate_ppgf = false) where ScalarGF
    P0 = ppgf.atomic_ppgf(grid, ed)
    dP0 = ppgf.initial_ppgf_derivative(ed, grid.contour.β)
    P = deepcopy(P0)
    P_orders = Vector{typeof(P)}()
      
    if interpolate_ppgf

        #P0 = [SplineInterpolatedGF(P0_s) for P0_s in P0]
        #P = [SplineInterpolatedGF(P_s, τ_max=grid[2]) for P_s in P]

        P0_interp = [IncSplineImaginaryTimeGF(P0_s, dP0_s) for (P0_s, dP0_s) in zip(P0, dP0)]
        P_interp = [IncSplineImaginaryTimeGF(P_s, dP0_s) for (P_s, dP0_s) in zip(P, dP0)]

        # -- Fill up P0 with all values
        for (s, p0_interp) in enumerate(P0_interp)
            τ_0 = grid[1]
            for τ in grid[2:end]
                extend!(p0_interp, P0[s][τ, τ_0])
            end
        end

        P0 = P0_interp
        P = P_interp

    end

    return new{ScalarGF, typeof(P0)}(ed, P0, P, P_orders, interaction_pairs, [])
  end
end

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
$(TYPEDEF)
"""
const Nodes = Vector{Node}

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
$(TYPEDEF)
"""
const NodePairs = Vector{NodePair}

"""
$(TYPEDSIGNATURES)

Returns a list of `Nodes` corresponding to the pair `pair::NodePair`.
"""
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

function get_node_idxs(n::Int, inch_node_pos::Int = nothing)::Vector{Int}
    if inch_node_pos !== nothing
        n_idxs = vcat(collect(n-1:-1:inch_node_pos+1), collect(inch_node_pos-1:-1:2))
    else
        n_idxs = collect((n - 1):-1:2)
    end
    return n_idxs
end

"""
$(TYPEDEF)

The `Configuration` struct defines a single diagram in a peudo-particle
expansion with fixed insertions of pseduo-particle interactions and
auxilliary operators.

$(TYPEDFIELDS)
"""
struct Configuration
    "List of nodes in time with associated operators"
    nodes::Nodes
    "List of pairs of nodes in time associated with a hybridization propagator"
    pairs::NodePairs
    "Parity of the diagram p = (-1)^N, with N number of hybridization line crossings"
    parity::Float64
    "List of groups of time nodes associated with expansion determinants"
    determinants::Determinants

    #"List of node operators"
    #operators::Vector{SectorBlockMatrix}

    "List of precomputed trace paths with operator sub matrices"
    paths::Vector{Vector{Tuple{Int, Int, Matrix{ComplexF64}}}}
    has_inch_node::Bool
    inch_node_idx::Int
    node_idxs::Vector{Int}

    function Configuration(single_nodes::Nodes, pairs::NodePairs, exp::Expansion)
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

        has_inch_node = any([ is_inch_node(node) for node in nodes ])
        paths = get_paths(exp, nodes)
        n_idxs = get_node_idxs(length(nodes), 3)

        parity = 1.0
        return new(nodes, pairs, parity, [], paths, has_inch_node, 3, n_idxs)
    end
    function Configuration(diagram::Diagram, exp::Expansion, d_bold::Int)

        contour = first(exp.P).grid.contour
        time = contour(0.0)
        n_f, n_w, n_i = Node(time), InchNode(time), Node(time)

        has_inch_node = d_bold > 0
        inch_node_idx = d_bold + 2
        single_nodes = has_inch_node ? [n_f, n_w, n_i] : [n_f, n_i]

        pairs = [ NodePair(time, time, diagram.pair_idxs[idx])
                  for (idx, (a, b)) in enumerate(diagram.topology.pairs) ]
        parity = (-1.0)^diag.n_crossings(diagram.topology)
        
        #@assert diag.parity(diagram.topology) == (-1.0)^diag.n_crossings(diagram.topology)
        #println("topology = $(diagram.topology), parity = $(parity)") # DEBUG

        n = diagram.topology.order*2
        pairnodes = [ Node(time) for i in 1:n ]

        for (idx, (f_idx, i_idx)) in enumerate(diagram.topology.pairs)
            p_idx = diagram.pair_idxs[idx]
            pairnodes[i_idx] = Node(time, OperatorReference(pair_flag, p_idx, 1))
            pairnodes[f_idx] = Node(time, OperatorReference(pair_flag, p_idx, 2))
        end

        reverse!(pairnodes)

        if length(pairnodes) > 0
            if has_inch_node
                nodes = vcat([n_i], pairnodes[1:d_bold], [n_w], pairnodes[d_bold+1:end], [n_f])
            else
                nodes = vcat([n_i], pairnodes, [n_f])
            end
        else
            nodes = single_nodes
        end

        paths = get_paths(exp, nodes)
        n_idxs = get_node_idxs(length(nodes), inch_node_idx)

        return new(nodes, pairs, parity, [], paths, has_inch_node, inch_node_idx, n_idxs)
    end
end

const Configurations = Vector{Configuration}

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

function eval(exp::Expansion, pairs::NodePairs, parity::Float64)

    order = length(pairs)
    val::ComplexF64 = parity * (-1)^order

    for pair in pairs
        #@assert pair.time_f >= pair.time_i
        val *= im * exp.pairs[pair.index].propagator(pair.time_f, pair.time_i)
    end
    return val
end

function set_bold_ppgf!(P::PPGF,
                        t_i::kd.TimeGridPoint,
                        t_f::kd.TimeGridPoint,
                        result::SectorBlockMatrix) where PPGF <: AllPPGFTypes
    for (s_i, (s_f, mat)) in result
        # Boldification must preserve the block structure
        @assert s_i == s_f
        P[s_i][t_f, t_i] = mat
    end
end

function set_bold_ppgf!(exp::Expansion{ScalarGF, Vector{IncSplineImaginaryTimeGF{ComplexF64, false}}},
                        t_i::kd.TimeGridPoint,
                        t_f::kd.TimeGridPoint,
                        result::SectorBlockMatrix) where ScalarGF <: kd.AbstractTimeGF{ComplexF64, true}
    for (s_i, (s_f, mat)) in result
        # Boldification must preserve the block structure
        @assert s_i == s_f
        extend!(exp.P[s_i], mat)
    end
end

function set_bold_ppgf!(exp::Expansion,
                        t_i::kd.TimeGridPoint,
                        t_f::kd.TimeGridPoint,
                        result::SectorBlockMatrix)
    for (s_i, (s_f, mat)) in result
        # Boldification must preserve the block structure
        @assert s_i == s_f
        exp.P[s_i][t_f, t_i] = mat
    end
end

# Specialization for spline-interpolated imaginary time PPGF
function set_bold_ppgf!(
    exp::Expansion{ScalarGF, Vector{SplineInterpolatedGF{ppgf.ImaginaryTimePPGFSector, ComplexF64, false}}},
    τ_i::kd.TimeGridPoint,
    τ_f::kd.TimeGridPoint,
    result::SectorBlockMatrix) where {ScalarGF <: kd.AbstractTimeGF{ComplexF64, true}}
    for (s_i, (s_f, mat)) in result
        # Boldification must preserve the block structure
        @assert s_i == s_f
        exp.P[s_i][τ_f, τ_i, τ_max = τ_f] = mat
    end
end

function Base.zeros(::Type{SectorBlockMatrix}, ed::ked.EDCore)
    Dict(s => (s, zeros(ComplexF64, length(sp), length(sp))) for (s, sp) in enumerate(ed.subspaces))
end

"""
$(TYPEDSIGNATURES)

Returns the [`SectorBlockMatrix`](@ref) representation of the many-body operator `op::Operator`.
"""
function operator_to_sector_block_matrix(exp::Expansion, op::Operator)::SectorBlockMatrix

    sbm = SectorBlockMatrix()

    op_blocks = ked.operator_blocks(exp.ed, op)

    for ((s_f, s_i), mat) in op_blocks
        sbm[s_i] = (s_f, mat)
    end

    return sbm
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
    elseif node.operator_ref.kind == identity_flag || is_inch_node(node)
        op = Operator(1.)
    else
        throw(BoundsError())
    end
    return operator_to_sector_block_matrix(exp, op)
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

function Base.:+(A::SectorBlockMatrix, B::SectorBlockMatrix)
    return merge((a,b) -> (a[1], a[2] + b[2]), A, B)
end

function Base.:-(A::SectorBlockMatrix, B::SectorBlockMatrix)
    return A + (-1) * B
end

function maxabs(A::SectorBlockMatrix)
    return mapreduce(x -> maximum(abs.(x[2])), max, values(A), init=-Inf)
end

function Base.zero(A::SectorBlockMatrix)
    Z = SectorBlockMatrix()
    for (s_i, (s_f, A_mat)) in A
        Z[s_i] = (s_f, zero(A_mat))
    end
    return Z
end

function Base.fill!(A::SectorBlockMatrix, x)
    for m in A
        fill!(m.second[2], x)
    end
end

function Base.isapprox(A::SectorBlockMatrix, B::SectorBlockMatrix; atol::Real=0)
    @assert keys(A) == keys(B)
    for k in keys(A)
        @assert A[k][1] == B[k][1]
        !isapprox(A[k][2], B[k][2], norm = mat -> norm(mat, Inf), atol=atol) && return false
    end
    return true
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

function eval(exp::Expansion, nodes::Nodes)

    node = first(nodes)
    val = operator(exp, node)

    P = exp.P # Inch with dressed ppsc propagator from start

    prev_node = node

    for (nidx, node) in enumerate(nodes[2:end])

        #if is_inch_node(prev_node)
        #    P = exp.P0 # Inch with bare ppsc propagator after inch time node
        #end

        op = operator(exp, node)
        P_interp = sector_block_matrix_from_ppgf(node.time, prev_node.time, P)

        val = (im * op * P_interp) * val

        prev_node = node
    end

    return -im * val
end

const Path = Vector{Tuple{Int, Int, Matrix{ComplexF64}}}
const Paths = Vector{Path}

function get_paths(exp::Expansion, nodes::Nodes)::Paths

    operators = [ operator(exp, node) for node in nodes ]
    N_sectors = length(exp.P)

    paths = Paths()
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

function eval(exp::Expansion, nodes::Nodes, paths::Vector{Vector{Tuple{Int, Int, Matrix{ComplexF64}}}}, has_inch_node::Bool)

    start = operator(exp, first(nodes))
    val = SectorBlockMatrix()

    for path in paths

        bold_P = has_inch_node
        prev_node = first(nodes)
        S_i, S_f, mat = first(path)

        for (nidx, node) in enumerate(nodes[2:end])

            #if is_inch_node(prev_node)
            #    bold_P = false
            #end

            s_i, s_f, op_mat = path[nidx + 1]

            #@assert node.time >= prev_node.time
            
            P_interp = bold_P ? exp.P[s_i](node.time, prev_node.time) : exp.P0[s_i](node.time, prev_node.time)

            mat = im * op_mat * P_interp * mat

            prev_node = node
        end
        val[S_i] = (S_f, -im * mat)
    end

    return val
end

function eval_acc!(val::SectorBlockMatrix, scalar::ComplexF64,
                   exp::Expansion, nodes::Nodes,
                   paths::Vector{Vector{Tuple{Int, Int, Matrix{ComplexF64}}}},
                   has_inch_node::Bool)

    start = operator(exp, first(nodes))

    @inbounds for path in paths

        bold_P = has_inch_node
        prev_node = first(nodes)
        S_i, S_f, mat = first(path)

        for (nidx, node) in enumerate(nodes[2:end])

            #if is_inch_node(prev_node)
            #    bold_P = false
            #end

            s_i, s_f, op_mat = path[nidx + 1]

            #@assert !(node.time < prev_node.time)
            #@assert node.time >= prev_node.time # BROKEN FIXME ?
            
            P_interp = bold_P ?
                exp.P[s_i](node.time, prev_node.time) :
                exp.P0[s_i](node.time, prev_node.time)

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
    return eval(exp, conf.pairs, conf.parity) * eval(exp, conf.nodes, conf.paths, conf.has_inch_node)
    #return eval(exp, conf.pairs) * eval(exp, conf.nodes)
end


function eval_acc!(value::SectorBlockMatrix, exp::Expansion, conf::Configuration)
    scalar::ComplexF64 = eval(exp, conf.pairs, conf.parity)
    eval_acc!(value, scalar, exp, conf.nodes, conf.paths, conf.has_inch_node)
    return
end

end # module configuration
