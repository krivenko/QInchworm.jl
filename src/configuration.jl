import PyCall; PyCall.pygui(:tk);
import PyPlot; const plt = PyPlot;

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

import QInchworm.ppgf

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

function sbm_from_ppgf(z2::Time, z1::Time, P::SectorGF)
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
        P_interp = sbm_from_ppgf(node.time, prev_node.time, P)
        val = (im * op * P_interp) * val

        prev_node = node
    end
        
    return -im * val
end

function eval(exp::Expansion, conf::Configuration)
  return eval(exp, conf.pairs) * eval(exp, conf.nodes)
end

@testset "node" begin

    β = 10.

    nt = 10
    ntau = 30
    tmax = 1.

    # -- Single state
    
    V = -0.1 # Hybridization
    μ = +0.1 # Chemical potential

    H = - μ * op.n("0")
    
    # -- Exact Diagonalization solver
    
    soi = KeldyshED.Hilbert.SetOfIndices([["0"]]);
    ed = KeldyshED.EDCore(H, soi)
    ρ = KeldyshED.density_matrix(ed, β)
    
    # -- Real-time Kadanoff-Baym contour
    
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
    grid = kd.FullTimeGrid(contour, nt, ntau);
    
    # -- Propagators
    
    ϵ = +0.1
    f = (t1, t2) -> -1.0im * (kd.heaviside(t1.bpoint, t2.bpoint)
                    - kd.fermi(ϵ, contour.β)) * exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ)
    
    g_bath = kd.FullTimeGF(f, grid, 1, kd.fermionic, true)
    
    Δ = deepcopy(g_bath)
    Δ.mat.data .*= V^2
    Δ.rm.data .*= V^2
    Δ.gtr.data .*= V^2
    Δ.les.data .*= V^2

    # -- Plotting
    if false
        τ = kd.imagtimes(grid)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.plot(τ, g_bath[:matsubara], label=plt.L"$g_\varepsilon(\tau)$")
        plt.xlabel(plt.L"$\tau$");
        plt.legend(loc="lower right");
        plt.ylim([-1., 0]);
        
        plt.subplot(1, 2, 2)
        plt.plot(τ, Δ[:matsubara], label=plt.L"$\Delta(\tau) = V^2 g_\varepsilon$")
        plt.xlabel(plt.L"$\tau$");
        plt.legend(loc="lower right");
        plt.ylim([-0.01, 0]);
        plt.tight_layout()
        
        plt.show()
    end
    
    ip_1 = InteractionPair(op.c("0"), op.c_dag("0"), Δ)

    ppsc_exp = Expansion(ed, grid, [ip_1])

    # -- Plot

    if false        
        plt.figure(figsize=(6, 4))
        τ = kd.imagtimes(grid)
        
        for (s, P0_s) in enumerate(ppsc_exp.P0)
            p0_s = P0_s[kd.imaginary_branch, kd.imaginary_branch]
            p0_s = vcat(p0_s[:, 1]...)
            plt.plot(τ, -imag(p0_s), "--", label="-Im[P0_$(s)]")
            plt.plot(τ, real(p0_s), "-k")
        end

        plt.xlabel(plt.L"$\tau$")
        plt.ylabel(plt.L"$\hat{G}(\tau)$");
        plt.legend(); 
        #plt.ylim([0, 1.1]);
        plt.grid(true)
        plt.show()
    end

    if false
        # -- Explicit configuration
        
        im_branch = contour[kd.imaginary_branch]

        zi = im_branch(0.0)
        zf = im_branch(1.0)
        zw = im_branch(0.5)

        ni = Node(zi)
        nf = Node(zf)
        nw = InchNode(zw)

        z1 = im_branch(0.2)
        z2 = im_branch(0.6)
        p1 = NodePair(z2, z1, 1)

        z3 = im_branch(0.4)
        z4 = im_branch(0.8)
        p2 = NodePair(z4, z3, 1)

        println("p1 = $p1")
        println("p2 = $p2")
        
        conf = Configuration([nf, nw, ni], [p1, p2])

        println("conf = $conf")
        println("---")
        println("conf.nodes = $(conf.nodes)")
        println("---")
        println("conf.pairs = $(conf.pairs)")
        
        val = eval(ppsc_exp, conf.pairs)
        println(val)

        vals = eval(ppsc_exp, conf.nodes)
        println(vals)
    end
    
    # -- 1st order inching
    
    tau_grid = grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    τ_beta = tau_grid[end]

    Δτ = -imag(tau_grid[2].bpoint.val - tau_grid[1].bpoint.val)

    ni = Node(τ_0.bpoint)
    
    for (fidx, τ_f) in enumerate(tau_grid[2:end])

        τ_w = tau_grid[fidx]

        println("fidx = $fidx, τ_f = $(τ_f)")

        nf = Node(τ_f.bpoint)
        nw = InchNode(τ_w.bpoint)

        conf0 = Configuration([nf, nw, ni], NodePairs())
        val = eval(ppsc_exp, conf0)

        for τ_1 in tau_grid[1:fidx]

            n1 = Node(τ_1.bpoint)
            
            begin
                p = NodePair(nf.time, n1.time, 1)
                conf = Configuration([nf, nw, ni], [p])
                val += + im * Δτ^2 * eval(ppsc_exp, conf)
            end
            
            begin
                p = NodePair(n1.time, nf.time, 1)
                conf = Configuration([nf, nw, ni], [p])
                val += - im * Δτ^2 * eval(ppsc_exp, conf)
            end

        end

        for (s, P_s) in enumerate(ppsc_exp.P)
            sf, mat = val[s]
            ppgf.set_matsubara(P_s, τ_f, mat)
        end

    end

    Z = ppgf.partition_function(ppsc_exp.P)
    λ = log(Z) / β

    ppgf.normalize(ppsc_exp.P, β);
    
        
    
    # -- Plot

    if true
        plt.figure(figsize=(6, 4))
        τ = kd.imagtimes(grid)
        
        for (s, P0_s) in enumerate(ppsc_exp.P0)
            p0_s = P0_s[kd.imaginary_branch, kd.imaginary_branch]
            p0_s = vcat(p0_s[:, 1]...)
            plt.plot(τ, -imag(p0_s), "--", label="-Im[P0_$(s)]")
            plt.plot(τ, real(p0_s), "-k")
        end

        for (s, P_s) in enumerate(ppsc_exp.P)
            p_s = P_s[kd.imaginary_branch, kd.imaginary_branch]
            p_s = vcat(p_s[:, 1]...)
            plt.plot(τ, -imag(p_s), "-", label="-Im[P_$(s)]")
            plt.plot(τ, real(p_s), "-k")
        end
        
        plt.xlabel(plt.L"$\tau$")
        plt.ylabel(plt.L"$\hat{G}(\tau)$");
        plt.legend(); 
        #plt.ylim([0, 1.1]);
        plt.grid(true)
        plt.show()
    end
    
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
