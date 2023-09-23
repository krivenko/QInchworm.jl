module topology_eval

#using DataStructures: DefaultDict
using Octavian

using TimerOutputs: TimerOutput, @timeit

using DocStringExtensions
using LinearAlgebra

using Keldysh; kd = Keldysh

using QInchworm
using QInchworm; cfg = QInchworm.configuration

using QInchworm: SectorBlockMatrix, SectorBlockMatrixReal, imag
using QInchworm.expansion: Expansion
using QInchworm.configuration: Configuration,
                               Time,
                               InteractionEnum,
                               pair_flag,
                               identity_flag,
                               inch_flag,
                               operator_flag

using QInchworm.diagrammatics: Topology,
                               is_doubly_k_connected,
                               generate_topologies,
                               Diagram


function get_topologies_at_order(order::Int64, k = nothing; with_external_arc = false)::Vector{Topology}
    topologies = generate_topologies(order)
    k === nothing && return topologies

    filter!(topologies) do top
        is_doubly_k_connected(top, k)
    end

    if with_external_arc
        topologies = [Topology(top.pairs, (-1)^k * top.parity) for top in topologies]
    end

    return topologies
end

""" Get all diagrams as combinations of a `Topology` and a list of pseudo particle interaction indicies

Parameters
----------

expansion : Pseudo particle expansion
order     : Inch worm perturbation order

Returns
-------

diagrams : Vector with tuples of topologies and pseudo particle interaction indicies

"""
function get_diagrams_at_order(
    expansion::Expansion, topologies::Vector{Topology}, order::Int64
    )::Vector{Diagram}

    # -- Generate all `order` lenght vector of combinations of pseudo particle interaction pair indices
    pair_idx_range = 1:length(expansion.pairs) # range of allowed pp interaction pair indices
    pair_idxs_combinations = collect(Iterators.product(repeat([pair_idx_range], outer=[order])...))

    diagrams = vec([ Diagram(topology, pair_idxs) for (topology, pair_idxs) in
            collect(Iterators.product(topologies, pair_idxs_combinations)) ])

    return diagrams
end

function get_configurations_and_diagrams_from_topologies(
    expansion::Expansion,
    #
    topologies::Vector{Topology}, order::Int64,
    #
    d_before::Union{Int, Nothing};
    op_pair_idx::Union{Int, Nothing} = nothing,
    return_configurations = true)::Tuple{Vector{Configuration}, Vector{Diagram}}

    r = rank_sub_range(length(topologies))
    rank_topologies = topologies[r]

    rank_diagrams = Diagram[]
    rank_configurations = Configuration[]

    for topology in rank_topologies

        # -- Generate all `order` lenght vector of combinations of pseudo particle interaction pair indices
        pair_idx_range = 1:length(expansion.pairs) # range of allowed pp interaction pair indices
        pair_idxs_combinations = Iterators.product(repeat([pair_idx_range], outer=[order])...)

        for pair_idxs in pair_idxs_combinations
            diagram = Diagram(topology, pair_idxs)

            if op_pair_idx === nothing
                configuration = Configuration(diagram, expansion, d_before)
            else
                configuration = Configuration(diagram, expansion, d_before, op_pair_idx)
            end

            if length(configuration.paths) > 0
                if return_configurations
                    push!(rank_configurations, configuration)
                end
                push!(rank_diagrams, diagram)
            end
        end
    end

    diagrams_out = all_gather(rank_diagrams)

    if return_configurations
        configurations = all_gather(rank_configurations)
    end

    if return_configurations
        return configurations, diagrams_out
    else
        return [], diagrams_out
    end
end

function get_configurations_and_diagrams_serial(
    expansion::Expansion,
    diagrams::Vector{Diagram},
    d_before::Union{Int, Nothing};
    op_pair_idx::Union{Int, Nothing} = nothing)::Tuple{Vector{Configuration}, Vector{Diagram}}

    diagrams_out = Diagram[]
    configurations = Configuration[]
    for (didx, diagram) in enumerate(diagrams)
        if op_pair_idx === nothing
            configuration = Configuration(diagram, expansion, d_before)
        else
            configuration = Configuration(diagram, expansion, d_before, op_pair_idx)
        end

        if length(configuration.paths) > 0
            push!(configurations, configuration)
            push!(diagrams_out, diagram)
        end
    end
    return configurations, diagrams_out
end

function get_configurations_and_diagrams(
    expansion::Expansion,
    diagrams::Vector{Diagram},
    d_before::Union{Int, Nothing};
    op_pair_idx::Union{Int, Nothing} = nothing,
    return_configurations = true)::Tuple{Vector{Configuration}, Vector{Diagram}}

    r = rank_sub_range(length(diagrams))
    rank_diagrams = diagrams[r]

    rank_diagrams_out = Diagram[]
    rank_configurations = Configuration[]
    for (didx, diagram) in enumerate(rank_diagrams)
        if op_pair_idx === nothing
            configuration = Configuration(diagram, expansion, d_before)
        else
            configuration = Configuration(diagram, expansion, d_before, op_pair_idx)
        end

        if length(configuration.paths) > 0
            if return_configurations
                push!(rank_configurations, configuration)
            end
            push!(rank_diagrams_out, diagram)
        end
    end

    diagrams_out = all_gather(rank_diagrams_out)
    if return_configurations
        configurations = all_gather(rank_configurations)
    end

    if return_configurations
        return configurations, diagrams_out
    else
        return [], diagrams_out
    end
end

function update_pair_node_times!(configuration::Configuration, diagram::Diagram, times::Vector{Time})
    for (t_idx, n_idx) in enumerate(configuration.pair_node_idxs)
        op_ref = configuration.nodes[n_idx].operator_ref
        configuration.nodes[n_idx] = cfg.Node(times[t_idx], op_ref)
    end

    for (p_idx, (idx_tf, idx_ti)) in enumerate(diagram.topology.pairs)
        int_idx = configuration.pairs[p_idx].index
        configuration.pairs[p_idx] = cfg.NodePair(times[idx_tf], times[idx_ti], int_idx)
    end
end

function eval(expansion::Expansion,
              diagrams::Vector{Diagram},
              configurations::Vector{Configuration},
              times::Vector{kd.BranchPoint},
    )::SectorBlockMatrix

    value = zeros(SectorBlockMatrix, expansion.ed)

    for (diagram, configuration) in zip(diagrams, configurations)
        update_pair_node_times!(configuration, diagram, times)
        cfg.eval_acc!(value, expansion, configuration)
    end

    return value
end

struct Node
    "Interaction type of operator"
    kind::InteractionEnum
    "Index for pair interaction arc"
    arc_index::Int64
    "Index to operator"
    operator_index::Int64
end

struct FixedNode
    "Reference to operator"
    node::Node
    "Contour time point"
    time::kd.BranchPoint
end

"""
$(TYPEDSIGNATURES)

Returns a fixed node at time `time` with an associated identity operator.
"""
function IdentityNode(time::kd.BranchPoint)::FixedNode
    return FixedNode(Node(identity_flag, -1, -1), time)
end

"""
$(TYPEDSIGNATURES)

Returns a fixed "inch" node at time `time` with an associated identity operator.
"""
function InchNode(time::kd.BranchPoint)::FixedNode
    return FixedNode(Node(inch_flag, -1, -1), time)
end

"""
$(TYPEDSIGNATURES)

Returns a fixed operator node at time `time` with an associated operator.
"""
function OperatorNode(time::kd.BranchPoint,
                      operator_pair_index::Int64,
                      operator_index::Int64)::FixedNode
    return FixedNode(Node(operator_flag, operator_pair_index, operator_index), time)
end

struct TopologyEvaluator
    "Pseudo-particle expansion problem"
    exp::Expansion

    "Configuration as a list of nodes arranged in the contour order"
    conf::Vector{Node}

    "Contour positions of nodes in the configuration"
    times::Vector{kd.BranchPoint}

    "Correspondence of node positions within a topology and a configuration"
    top_to_conf_pos::Vector{Int64}

    "Must the bold PPGFs be used?"
    use_bold_prop::Bool

    """
    PPGFs evaluated at all relevant pairs of time arguments.

    ppgf_mats[i, s] is the s-th diagonal block of exp.P (or exp.P0) evaluated at the pair
    of time points ``(t_{i+1}, t_i)``.
    """
    ppgf_mats::Array{Matrix{ComplexF64}, 2}
    ppgf_mats_real::Array{Matrix{Float64}, 2}

    """
    Pair interaction arcs evaluated at all relevant pairs of time arguments.

    pair_ints[a, p] is the propagator from `exp.pairs[p]` evaluated at the pair of time
    points corresponding to the a-th arc in a topology.
    """
    pair_ints::Array{ComplexF64, 2}
    pair_ints_real::Array{Float64, 2}

    """
    Indices of pair interactions within `exp.pairs` assigned to each
    interaction arc in a topology.
    """
    selected_pair_ints::Vector{Int64}

    """Pre-allocated container for per-topology evaluation results"""
    top_result::SectorBlockMatrix

    """ Internal performance timer"""
    tmr::TimerOutput

    """ Temporary ppgf_weight storage (for reducing allocations)"""
    tmp_mv::Array{ComplexF64, 2}
    tmp_mv_real::Array{Float64, 2}

    tmp1::Array{ComplexF64, 1}
    tmp2::Array{ComplexF64, 1}

    tmp1_real::Array{Float64, 1}
    tmp2_real::Array{Float64, 1}
    
    #tmp1::Array{Matrix{Float64}, 2}
    #tmp2::Array{Matrix{Float64}, 2}
    
    #tmp1::Array{Matrix{ComplexF64}, 2}
    #tmp2::Array{Matrix{ComplexF64}, 2}

    tmp_idx::Ref{Int64}

    identity_mats_real::Array{Matrix{Float64}, 1}

    "Block matrix representation of paired operators (operator_i, operator_f)"
    pair_operator_mat::Vector{Tuple{SectorBlockMatrix, SectorBlockMatrix}}
    pair_operator_mat_real::Vector{Tuple{SectorBlockMatrixReal, SectorBlockMatrixReal}}
    "Block matrix representation of corr_operators"
    corr_operators_mat::Vector{Tuple{SectorBlockMatrix, SectorBlockMatrix}}
    corr_operators_mat_real::Vector{Tuple{SectorBlockMatrixReal, SectorBlockMatrixReal}}

    #""" Histogram over matrix sizes """
    #matrix_sizes::DefaultDict{Tuple{Int64, Int64, Int64}, Int64}
    
    function TopologyEvaluator(exp::Expansion,
                               order::Int,
                               fixed_nodes::Dict{Int, FixedNode};
                               tmr::TimerOutput = TimerOutput())
        n_nodes = 2 * order + length(fixed_nodes)
        @assert maximum(keys(fixed_nodes)) <= n_nodes

        # Prepare a skeleton of the configuration by placing only the fixed nodes
        conf = Vector{Node}(undef, n_nodes)
        times = Vector{kd.BranchPoint}(undef, n_nodes)

        use_bold_prop = false
        for (pos, fn) in fixed_nodes
            conf[pos] = fn.node

            if fn.node.kind == inch_flag || fn.node.kind == operator_flag
                use_bold_prop = true
            end

            # Copy time of fixed nodes to `times`
            times[pos] = fixed_nodes[pos].time
        end

        # Build the `top_to_conf_pos` map.
        # We need the reverse() here because the orders of nodes in a topology and in a
        # configurations are reversed.
        top_to_conf_pos = [pos for pos in reverse(1:n_nodes) if !haskey(fixed_nodes, pos)]

        ppgf_mats = [Matrix{ComplexF64}(undef, norbitals(p), norbitals(p))
                     for _ in 1:(n_nodes-1), p in exp.P]
        ppgf_mats_real = [Matrix{Float64}(undef, norbitals(p), norbitals(p))
                     for _ in 1:(n_nodes-1), p in exp.P]

        pair_ints = Array{ComplexF64}(undef, order, length(exp.pairs))
        pair_ints_real = Array{Float64}(undef, order, length(exp.pairs))

        selected_pair_ints = Vector{Int64}(undef, order)

        top_result = zeros(SectorBlockMatrix, exp.ed)

        # move allocation to exp
        m = maximum([ norbitals(p) for p in exp.P ])

        tmp_mv = Array{ComplexF64, 2}(undef, m*m, n_nodes + 1)
        tmp_mv_real = Array{Float64, 2}(undef, m*m, n_nodes + 1)

        tmp1 = Array{ComplexF64, 1}(undef, m*m)
        tmp2 = Array{ComplexF64, 1}(undef, m*m)

        tmp1_real = Array{Float64, 1}(undef, m*m)
        tmp2_real = Array{Float64, 1}(undef, m*m)
        
        #tmp1 = [Matrix{ComplexF64}(undef, i, j ) for i in 1:m, j in 1:m]
        #tmp2 = [Matrix{ComplexF64}(undef, i, j ) for i in 1:m, j in 1:m]

        #tmp1 = [Matrix{Float64}(undef, i, j ) for i in 1:m, j in 1:m]
        #tmp2 = [Matrix{Float64}(undef, i, j ) for i in 1:m, j in 1:m]
        
        tmp_idx = 1

        identity_mats_real = [ Matrix{Float64}(UniformScaling(1), norbitals(p), norbitals(p)) for p in exp.P ]
        
        pair_operator_mat = [ pair for pair in exp.pair_operator_mat ]
        corr_operators_mat = [ pair for pair in exp.corr_operators_mat ]

        pair_operator_mat_real = [ real.(pair) for pair in exp.pair_operator_mat ]
        corr_operators_mat_real = [ real.(pair) for pair in exp.corr_operators_mat ]

        #matrix_sizes = DefaultDict{Tuple{Int64, Int64, Int64}, Int64}(0)
        
        return new(exp,
                   conf,
                   times,
                   top_to_conf_pos,
                   use_bold_prop,
                   ppgf_mats,
                   ppgf_mats_real,
                   pair_ints,
                   pair_ints_real,
                   selected_pair_ints,
                   top_result,
                   #tmr, tmp_mv, tmp1, tmp2)
                   #tmr, tmp_mv, tmp1, tmp2, tmp_idx,
                   tmr, tmp_mv, tmp_mv_real, tmp1, tmp2, tmp1_real, tmp2_real, tmp_idx,
                   #pair_operator_mat, corr_operators_mat
                   identity_mats_real,
                   pair_operator_mat, pair_operator_mat_real, corr_operators_mat, corr_operators_mat_real
                   #matrix_sizes
                   )
    end
end

function (eval::TopologyEvaluator)(topology::Topology,
    times::Vector{kd.BranchPoint})::SectorBlockMatrix
    return eval([topology], times)
end

function (eval::TopologyEvaluator)(topologies::Vector{Topology},
                                   times::Vector{kd.BranchPoint})::SectorBlockMatrix

    @timeit eval.tmr "times" begin
        
    # Update eval.times
    for (pos, t) in zip(eval.top_to_conf_pos, times)
        eval.times[pos] = t
    end

    end; @timeit eval.tmr "ppgf_mats" begin

    # Pre-compute eval.ppgf_mats
    for i in axes(eval.ppgf_mats, 1)
        time_i = eval.times[i]
        time_f = eval.times[i + 1]

        # Tackle time ordering violations caused by rounding errors
        if time_f < time_i
            time_f = time_i
        end

        for s in axes(eval.ppgf_mats, 2)
            if eval.use_bold_prop
                #eval.ppgf_mats[i, s] = im * eval.exp.P[s](time_f, time_i)
                kd.interpolate!(eval.ppgf_mats[i, s], eval.exp.P[s], time_f, time_i)
            else
                #eval.ppgf_mats[i, s] = im * eval.exp.P0[s](time_f, time_i)
                kd.interpolate!(eval.ppgf_mats[i, s], eval.exp.P0[s], time_f, time_i)
            end
            eval.ppgf_mats[i, s] .*= im
            eval.ppgf_mats_real[i, s] .= real(eval.ppgf_mats[i, s])
        end
    end

    end # tmr
    #end; @timeit eval.tmr "eval topologies loop" begin
        
    result = zeros(SectorBlockMatrix, eval.exp.ed)

    for top in topologies # TODO: Parallelization opportunity I

        @timeit eval.tmr "pair_ints" begin

        @assert length(times) == 2 * length(top.pairs)

        # Pre-compute eval.pair_ints and place pair interaction nodes into the configuration
        for (a, arc) in enumerate(top.pairs)
            pos_head = eval.top_to_conf_pos[arc[2]]
            pos_tail = eval.top_to_conf_pos[arc[1]]
            @assert pos_tail > pos_head

            eval.conf[pos_head] = Node(pair_flag, a, 1)
            eval.conf[pos_tail] = Node(pair_flag, a, 2)

            time_i = eval.times[pos_head]
            time_f = eval.times[pos_tail]
            # Tackle time ordering violations caused by rounding errors
            if time_f < time_i
                time_f = time_i
            end

            for (p, int_pair) in enumerate(eval.exp.pairs)
                eval.pair_ints[a, p] = im * int_pair.propagator(time_f, time_i)
                eval.pair_ints_real[a, p] = real.(eval.pair_ints[a, p])
            end
        end

        end;

        if true
        @timeit eval.tmr "tree (cplx opt)" begin
                
        fill!(eval.top_result, 0.0)

        # Traverse the configuration tree for each initial subspace
        for s_i in eachindex(eval.exp.P) # TODO: Parallelization opportunity II
            _traverse_configuration_tree_opt_cplx!(eval,
                                          view(eval.conf, :),
                                          s_i, s_i,
                                          eval.exp.identity_mat[s_i][2],
                                          ComplexF64(1))
        end

        top_result_cplx = deepcopy(eval.top_result)
            
        end; # tmr
            
        end # if 
        
        if true

        @timeit eval.tmr "tree (real opt)" begin

        fill!(eval.top_result, 0.0)

        # Traverse the configuration tree for each initial subspace
        for s_i in eachindex(eval.exp.P) # TODO: Parallelization opportunity II

            _traverse_configuration_tree_opt_real!(eval,
                                              view(eval.conf, :),
                                              s_i, s_i,
                                              eval.identity_mats_real[s_i],
                                              Float64(1))
        end
                
        top_result_real = deepcopy(eval.top_result)

        end # tree trav tmr

        end

        if true
        @timeit eval.tmr "tree" begin
                
        fill!(eval.top_result, 0.0)

        # Traverse the configuration tree for each initial subspace
        for s_i in eachindex(eval.exp.P) # TODO: Parallelization opportunity II
            _traverse_configuration_tree!(eval,
                                          view(eval.conf, :),
                                          s_i, s_i,
                                          eval.exp.identity_mat[s_i][2],
                                          ComplexF64(1))
        end

        top_result_ref = deepcopy(eval.top_result)
            
        end; # tmr
            
        end # if 
        
        diff_real = maximum(abs(top_result_ref - top_result_real))
        diff_cplx = maximum(abs(top_result_ref - top_result_cplx))
        
        if diff_real > 1e-9
            @show diff_real
            #@show top_result_ref
            #@show top_result_real
            #exit()
        end

        if diff_cplx > 1e-9
            @show diff_cplx
        end

        #@show eval.matrix_sizes
        #for key in keys(eval.matrix_sizes)
        #    delete!(eval.matrix_sizes, key)
        #end
        
        if false
        diff = maximum(abs(top_result_ref - eval.top_result))
        #if !(top_result_ref ≈ eval.top_result)
        if diff > 1e-9
            #@show top_result_ref
            #@show eval.top_result
            #@show top_result_ref - eval.top_result
            @show diff
        end
        #@assert top_result_ref ≈ eval.top_result
        end # if
        
        result += -im * top.parity * (-1)^top.order * eval.top_result
    end

    #end; # tmr
    
    return result
end

function _traverse_configuration_tree!(eval::TopologyEvaluator,
                                       conf::SubArray{Node, 1},
                                       s_i::Int64,
                                       s_f::Int64,
                                       ppgf_weight::Matrix{ComplexF64},
                                       pair_int_weight::ComplexF64)

    # Are we at a leaf?
    if isempty(conf)
        if s_i == s_f # Is the resulting configuration block-diagonal?
            val = pair_int_weight * ppgf_weight
            eval.top_result[s_i] = (s_f, eval.top_result[s_i][2] + val)
        end
        return
    end

    # Current position within the configuration
    pos = length(parent(conf)) - length(conf) + 1

    node = conf[1]                  # Current node
    conf_tail = @view conf[2:end]   # The rest of the configuration

    if node.kind == pair_flag

        if node.operator_index == 1 # Head of an interaction arc

            # Loop over all interaction pairs attachable to this node
            for int_index in eval.exp.subspace_attachable_pairs[s_i]

                # Select an interaction for this arc
                eval.selected_pair_ints[node.arc_index] = int_index

                s_next, mat = eval.exp.pair_operator_mat[int_index][1][s_i]
                ppgf_weight_next = (pos == 1) ?
                                   mat * ppgf_weight :
                                   mat * eval.ppgf_mats[pos - 1, s_i] * ppgf_weight

                _traverse_configuration_tree!(eval,
                                              conf_tail,
                                              s_next, s_f,
                                              ppgf_weight_next,
                                              pair_int_weight)
            end

        else # Tail of an interaction arc

            int_index = eval.selected_pair_ints[node.arc_index]

            op_sbm = eval.exp.pair_operator_mat[int_index][2]
            if haskey(op_sbm, s_i)

                s_next, mat = op_sbm[s_i]
                ppgf_weight_next = (pos == 1) ?
                                   mat * ppgf_weight :
                                   mat * eval.ppgf_mats[pos - 1, s_i] * ppgf_weight

                pair_int_weight_next =
                    eval.pair_ints[node.arc_index, int_index] * pair_int_weight

                _traverse_configuration_tree!(eval,
                                              conf_tail,
                                              s_next, s_f,
                                              ppgf_weight_next,
                                              pair_int_weight_next)
            end

        end

    elseif node.kind == operator_flag

        op_sbm = eval.exp.corr_operators_mat[node.arc_index][node.operator_index]
        if haskey(op_sbm, s_i)
            s_next, op_mat = op_sbm[s_i]
            ppgf_weight_next = (pos == 1) ?
                               op_mat * ppgf_weight :
                               op_mat * eval.ppgf_mats[pos - 1, s_i] * ppgf_weight
            _traverse_configuration_tree!(eval,
                                          conf_tail,
                                          s_next,
                                          s_f,
                                          ppgf_weight_next,
                                          pair_int_weight)
        end

    elseif node.kind ∈ (identity_flag, inch_flag)

        ppgf_weight_next = (pos == 1) ?
                           ppgf_weight :
                           eval.ppgf_mats[pos - 1, s_i] * ppgf_weight
        _traverse_configuration_tree!(eval,
                                      conf_tail,
                                      s_i, s_f,
                                      ppgf_weight_next,
                                      pair_int_weight)

    else
        @assert false
    end

end

function _traverse_configuration_tree_opt_cplx!(eval::TopologyEvaluator,
                                       conf::SubArray{Node, 1},
                                       s_i::Int64,
                                       s_f::Int64,
                                       ppgf_weight::AbstractMatrix{ComplexF64},
                                       pair_int_weight::ComplexF64)

    @inline function mat_view(vec, n, m)
        return @inbounds reshape(view(vec, 1:n*m), n, m)
    end
    
    @inline function mygemm!(C, A, B)
        @inbounds @fastmath for m ∈ axes(A,1), n ∈ axes(B,2)
            Cmn = zero(eltype(C))
            for k ∈ axes(A,2)
                Cmn += A[m,k] * B[k,n]
            end
            C[m,n] = Cmn
        end
    end

    @inline function matmul_prealloc(
        A::Matrix{ComplexF64}, B::AbstractMatrix{ComplexF64}, eval::TopologyEvaluator)

        #@timeit eval.tmr "matmul_prealloc" begin

        tmp = (parent(parent(B)) === eval.tmp1) ? eval.tmp2 : eval.tmp1
        C = mat_view(tmp, size(A, 1), size(B, 2))

        #if eval.tmp_idx[] == 1
        #    tmp = eval.tmp2
        #    eval.tmp_idx[] = 2
        #else
        #    tmp = eval.tmp1
        #    eval.tmp_idx[] = 1
        #end
        #C = @inbounds tmp[size(A, 1), size(B, 2)]
        
        #mul!(C, A, B)
        mygemm!(C, A, B)
        #Octavian.matmul!(C, A, B)

        #end
        return C
    end
    
    while !isempty(conf)

        #@timeit eval.tmr "pos, node, conf" begin

        # Current position within the configuration
        pos::Int64 = length(parent(conf)) - length(conf) + 1
        node = @inbounds conf[1]            # Current node
        conf = @inbounds @view conf[2:end]  # The rest of the configuration

        #end # tmr
        
        #@timeit eval.tmr "ppgf_mat apply" begin
        if pos > 1
            #ppgf_weight = (im * eval.ppgf_mats[pos - 1, s_i]) * ppgf_weight
            #ppgf_weight = @inbounds matmul_prealloc(eval.ppgf_mats[pos - 1, s_i], ppgf_weight, eval)
            ppgf_mat = @inbounds eval.ppgf_mats[pos - 1, s_i]
            ppgf_weight = matmul_prealloc(ppgf_mat, ppgf_weight, eval)
        end
        #end # tmr
        
        if node.kind == pair_flag
            
            if node.operator_index == 1 # Head of an interaction arc

                #@timeit eval.tmr "ppgf_weight store" begin
                # ppgf_weight needs separate storage (since the _travers... calls use tmp1 & tmp2)
                ppgf_tmp = @inbounds mat_view(view(eval.tmp_mv, :, pos), size(ppgf_weight, 1), size(ppgf_weight, 2))
                ppgf_tmp .= ppgf_weight
                #ppgf_tmp = copy(ppgf_weight)
                #end # tmr
                
                # Loop over all interaction pairs attachable to this node
                for int_index in @inbounds eval.exp.subspace_attachable_pairs[s_i]
                    
                    #@timeit eval.tmr "prep recursion" begin
                        
                    # Select an interaction for this arc
                    @inbounds eval.selected_pair_ints[node.arc_index] = int_index
                    
                    s_next, mat = @inbounds eval.exp.pair_operator_mat[int_index][1][s_i]
                    #s_next, mat = @inbounds eval.pair_operator_mat[int_index][1][s_i]
                    #s_next, mat = @inbounds eval.pair_operator_mat_real[int_index][1][s_i]
                    
                    #ppgf_weight_next = mat * ppgf_weight
                    ppgf_weight_next = matmul_prealloc(mat, ppgf_tmp, eval)

                    #end
                    
                    #@timeit eval.tmr "recursion" begin
                    _traverse_configuration_tree_opt_cplx!(
                        eval, conf, s_next, s_f, ppgf_weight_next, pair_int_weight)
                    #end
                end

                return # Have to return here to terminate the eval

            else # Tail of an interaction arc
                
                #@timeit eval.tmr "arc tail" begin
                    
                int_index = @inbounds eval.selected_pair_ints[node.arc_index]
                op_sbm = @inbounds eval.exp.pair_operator_mat[int_index][2]
                #op_sbm = @inbounds eval.pair_operator_mat[int_index][2]
                #op_sbm = @inbounds eval.pair_operator_mat_real[int_index][2]
                haskey(op_sbm, s_i) || return
                s_i, mat = op_sbm[s_i]

                #pair_int_weight *= @inbounds eval.pair_ints[node.arc_index, int_index]
                pair_int_weight *= @inbounds eval.pair_ints_real[node.arc_index, int_index]

                #ppgf_weight = mat * ppgf_weight
                ppgf_weight = matmul_prealloc(mat, ppgf_weight, eval)
                #end # tmr
                
            end

        elseif node.kind == operator_flag

            #@timeit eval.tmr "corr op" begin
            op_sbm = @inbounds eval.exp.corr_operators_mat[node.arc_index][node.operator_index]
            #op_sbm = @inbounds eval.corr_operators_mat[node.arc_index][node.operator_index]
            #op_sbm = @inbounds eval.corr_operators_mat_real[node.arc_index][node.operator_index]
            haskey(op_sbm, s_i) || return
            s_i, mat = op_sbm[s_i]
            
            #ppgf_weight = mat * ppgf_weight
            ppgf_weight = matmul_prealloc(mat, ppgf_weight, eval)
            
            #end # tmr
        end
        
    end

    #@timeit eval.tmr "assign" begin
    # We are at a leaf
    #@assert s_i == s_f
    @inbounds eval.top_result[s_i][2] .+= pair_int_weight .* ppgf_weight
    #end # tmr

end

function _traverse_configuration_tree_opt_real!(eval::TopologyEvaluator,
                                       conf::SubArray{Node, 1},
                                       s_i::Int64,
                                       s_f::Int64,
                                       ppgf_weight::AbstractMatrix{Float64},
                                       pair_int_weight::Float64)
                                       #ppgf_weight::AbstractMatrix,
                                       ##ppgf_weight::AbstractMatrix{ComplexF64},
                                       #pair_int_weight::ComplexF64)

    @inline function mat_view(vec, n, m)
        return @inbounds reshape(view(vec, 1:n*m), n, m)
    end
    
    @inline function mygemm!(C, A, B)
        @inbounds @fastmath for m ∈ axes(A,1), n ∈ axes(B,2)
            Cmn = zero(eltype(C))
            for k ∈ axes(A,2)
                Cmn += A[m,k] * B[k,n]
            end
            C[m,n] = Cmn
        end
    end

    @inline function matmul_prealloc(
        #A, B, eval::TopologyEvaluator)
        A::Matrix{Float64}, B::AbstractMatrix{Float64}, eval::TopologyEvaluator)
        #A::Matrix{Float64}, B::Matrix{Float64}, eval::TopologyEvaluator)::Matrix{Float64}
        #A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, eval::TopologyEvaluator)
        #A::Matrix{ComplexF64}, B::AbstractMatrix{ComplexF64}, eval::TopologyEvaluator)

        #@timeit eval.tmr "matmul_prealloc" begin

        tmp = (parent(parent(B)) === eval.tmp1_real) ? eval.tmp2_real : eval.tmp1_real
        C = mat_view(tmp, size(A, 1), size(B, 2))

        #if eval.tmp_idx[] == 1
        #    tmp = eval.tmp2
        #    eval.tmp_idx[] = 2
        #else
        #    tmp = eval.tmp1
        #    eval.tmp_idx[] = 1
        #end
        #C = @inbounds tmp[size(A, 1), size(B, 2)]
        
        #mul!(C, A, B)
        #mygemm!(C, A, B)
        Octavian.matmul!(C, A, B)

        #eval.matrix_sizes[(size(A, 1), size(A, 2), size(B, 2))] += 1
        
        #end
        return C
    end
    
    while !isempty(conf)

        #@timeit eval.tmr "pos, node, conf" begin

        # Current position within the configuration
        pos::Int64 = length(parent(conf)) - length(conf) + 1
        node = @inbounds conf[1]            # Current node
        conf = @inbounds @view conf[2:end]  # The rest of the configuration

        #end # tmr
        
        #@timeit eval.tmr "ppgf_mat apply" begin
        if pos > 1
            #ppgf_weight = (im * eval.ppgf_mats[pos - 1, s_i]) * ppgf_weight
            #ppgf_weight = @inbounds matmul_prealloc(eval.ppgf_mats[pos - 1, s_i], ppgf_weight, eval)
            ppgf_mat = @inbounds eval.ppgf_mats_real[pos - 1, s_i]
            ppgf_weight = matmul_prealloc(ppgf_mat, ppgf_weight, eval)
        end
        #end # tmr
        
        if node.kind == pair_flag
            
            if node.operator_index == 1 # Head of an interaction arc

                #@timeit eval.tmr "ppgf_weight store" begin
                # ppgf_weight needs separate storage (since the _travers... calls use tmp1 & tmp2)
                ppgf_tmp = @inbounds mat_view(view(eval.tmp_mv_real, :, pos), size(ppgf_weight, 1), size(ppgf_weight, 2))
                ppgf_tmp .= ppgf_weight
                #ppgf_tmp = copy(ppgf_weight)
                #end # tmr
                
                # Loop over all interaction pairs attachable to this node
                for int_index in @inbounds eval.exp.subspace_attachable_pairs[s_i]
                    
                    #@timeit eval.tmr "prep recursion" begin
                        
                    # Select an interaction for this arc
                    @inbounds eval.selected_pair_ints[node.arc_index] = int_index
                    
                    #s_next, mat = @inbounds eval.exp.pair_operator_mat[int_index][1][s_i]
                    #s_next, mat = @inbounds eval.pair_operator_mat[int_index][1][s_i]
                    s_next, mat = @inbounds eval.pair_operator_mat_real[int_index][1][s_i]

                    #@show typeof(mat)
                    #@show mat
                    #exit()
                    
                    #ppgf_weight_next = mat * ppgf_weight
                    ppgf_weight_next = matmul_prealloc(mat, ppgf_tmp, eval)

                    #end
                    
                    #@timeit eval.tmr "recursion" begin
                    _traverse_configuration_tree_opt_real!(
                        eval, conf, s_next, s_f, ppgf_weight_next, pair_int_weight)
                    #end
                end

                return # Have to return here to terminate the eval

            else # Tail of an interaction arc
                
                #@timeit eval.tmr "arc tail" begin
                    
                int_index = @inbounds eval.selected_pair_ints[node.arc_index]
                #op_sbm = @inbounds eval.exp.pair_operator_mat[int_index][2]
                #op_sbm = @inbounds eval.pair_operator_mat[int_index][2]
                op_sbm = @inbounds eval.pair_operator_mat_real[int_index][2]
                haskey(op_sbm, s_i) || return
                s_i, mat = op_sbm[s_i]

                #pair_int_weight *= @inbounds eval.pair_ints[node.arc_index, int_index]
                pair_int_weight *= @inbounds eval.pair_ints_real[node.arc_index, int_index]

                #ppgf_weight = mat * ppgf_weight
                ppgf_weight = matmul_prealloc(mat, ppgf_weight, eval)
                #end # tmr
                
            end

        elseif node.kind == operator_flag

            #@timeit eval.tmr "corr op" begin
            #op_sbm = @inbounds eval.exp.corr_operators_mat[node.arc_index][node.operator_index]
            #op_sbm = @inbounds eval.corr_operators_mat[node.arc_index][node.operator_index]
            op_sbm = @inbounds eval.corr_operators_mat_real[node.arc_index][node.operator_index]
            haskey(op_sbm, s_i) || return
            s_i, mat = op_sbm[s_i]
            
            #ppgf_weight = mat * ppgf_weight
            ppgf_weight = matmul_prealloc(mat, ppgf_weight, eval)
            
            #end # tmr
        end
        
    end

    #@timeit eval.tmr "assign" begin
    # We are at a leaf
    #@assert s_i == s_f
    @inbounds eval.top_result[s_i][2] .+= pair_int_weight .* ppgf_weight
    #end # tmr

end

end # module topology_eval
