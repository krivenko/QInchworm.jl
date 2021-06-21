using Test

import LinearAlgebra; la = LinearAlgebra

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

const Time = kd.BranchPoint
const Times = Vector{Time}

const ScalarGF = kd.FullTimeGF{ComplexF64, true}
const MatrixGF = kd.GenericTimeGF{ComplexF64, false}
const SectorGF = Vector{MatrixGF}

const Operator = op.RealOperatorExpr
const Operators = Vector{Operator}

const OperatorBlocks = Dict{Tuple{Int64, Int64}, Matrix{Float64}}
const SectorBlockMatrix = Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}

import QInchworm.ppgf: atomic_ppgf

@enum InteractionEnum pair_flag=1 determinant_flag=2 identity_flag=3

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
    P0 = atomic_ppgf(grid, ed)
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

const Nodes = Vector{Node}

struct NodePair
  time_f::Time
  time_i::Time
  index::Int64
end

const NodePairs = Vector{NodePair}

function Nodes(pair::NodePair, idx::Int64)
    n1 = Node(pair.time_i, OperatorReference(pair_flag, idx, 1))
    n2 = Node(pair.time_f, OperatorReference(pair_flag, idx, 2))
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

        for (idx, pair) in enumerate(pairs)
            append!(nodes, Nodes(pair, idx))
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

function eval(exp::Expansion, pairs::NodePairs)
    val::ComplexF64 = 1.
    for pair in pairs
        val *= exp.pairs[pair.index].propagator(pair.time_f, pair.time_i)
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
    elseif node.operator_ref.kind == identity_flag
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

function sbm_from_ppgf(z2::Time, z1::Time, P::SectorGF)
    M = SectorBlockMatrix()
    for (sidx, p) in enumerate(P)
        M[sidx] = (sidx, p(z2, z1))
    end
    return M
end

function eval(exp::Expansion, nodes::Nodes)

    println("------")
    node = first(nodes)
    val = operator(exp, node)
    println(val)

    prev_node = node
    for (nidx, node) in enumerate(nodes[2:end])
        println("------")
        println("node = $node")

        z1 = prev_node.time
        z2 = node.time
        println("z1 = $z1")
        println("z2 = $z2")

        op = operator(exp, node)
        println("op = $op")

        p_val = sbm_from_ppgf(z2, z1, exp.P0)
        println("p_val = $p_val")
        
        val = op * p_val * val
        println(val)

        prev_node = node
    end
    println("======")
        
    return val
end

function eval(exp::Expansion, conf::Configuration)
  return eval(exp, conf.pairs) * eval(exp, conf.nodes)
end

@testset "node" begin

    β = 10.

    U = +2.0 # Local interaction
    V = -0.1 # Hybridization
    B = +0.0 # Magnetic field
    μ = -0.1 # Chemical potential

    nt = 10
    ntau = 10

    # Hubbard-atom Hamiltonian

    H = U * (op.n("up") - 1/2) * (op.n("do") - 1/2) 
    H += V * (op.c_dag("up") * op.c("do") + op.c_dag("do") * op.c("up")) 
    H += B * (op.n("up") - op.n("do"))
    H += μ * (op.n("up") + op.n("do"))

    # Exact Diagonalization solver
    
    soi = KeldyshED.Hilbert.SetOfIndices([["up"], ["do"]]);
    ed = KeldyshED.EDCore(H, soi)
    ρ = KeldyshED.density_matrix(ed, β)
    
    # Real-time Kadanoff-Baym contour
    
    contour = kd.twist(kd.FullContour(tmax=30., β=β));
    grid = kd.FullTimeGrid(contour, nt, ntau);
    
    # Single particle Green's function
    
    u = KeldyshED.Hilbert.IndicesType(["up"])
    d = KeldyshED.Hilbert.IndicesType(["do"])

    # -- Propagators
    
    ϵ = -1.0
    f = (t1, t2) -> -1.0im * (kd.heaviside(t1.bpoint, t2.bpoint)
                    - kd.fermi(ϵ, contour.β)) * exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ)

    Δ = kd.FullTimeGF(f, grid, 1, kd.fermionic, true)

    ip_1 = InteractionPair(op.c("up"), op.c_dag("up"), Δ)
    ip_2 = InteractionPair(op.c("do"), op.c_dag("do"), Δ)

    ppsc_exp = Expansion(ed, grid, [ip_1, ip_2])

    # -- Explicit configuration
    
    im_branch = contour[kd.imaginary_branch]

    zi = im_branch(0.0)
    zf = im_branch(1.0)
    ni = Node(zi)
    nf = Node(zf)

    z1 = im_branch(0.2)
    z2 = im_branch(0.6)
    p1 = NodePair(z2, z1, 1)

    z3 = im_branch(0.4)
    z4 = im_branch(0.8)
    p2 = NodePair(z4, z3, 2)
    
    conf = Configuration([nf, ni], [p1, p2])

    val = eval(ppsc_exp, conf.pairs)
    println(val)

    vals = eval(ppsc_exp, conf.nodes)
    println(vals)
    
    # --
    if false
        im_branch = contour[kd.imaginary_branch]
    
        zi = im_branch(0.)
        zf = im_branch(1.)
        println(zi)
        println(zf)

        z1 = im_branch(0.2)
        z2 = im_branch(0.8)

        int_idx = 1
        op_idx = 1
        op_ref = OperatorReference(pair, int_idx, op_idx)
        println(op_ref)
        
        n1 = Node(z1, op_ref)
        n2 = Node(z2, op_ref)
        println(n1)
        println(n2)
        
        ip_idx = 1
        p = NodePair(z2, z1, ip_idx)
    end
    
end
