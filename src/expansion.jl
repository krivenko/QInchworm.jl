module expansion

using DocStringExtensions

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm: SectorBlockMatrix, operator_to_sector_block_matrix
using QInchworm.ppgf
using QInchworm.diagrammatics: Topology, Diagram
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.spline_gf: IncSplineImaginaryTimeGF, extend!

const Operator = op.RealOperatorExpr

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

Base.zero(P::T) where {T <: AllPPGFTypes} = [zero(p) for p in P]

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

struct InteractionDeterminant{PPGFSector <: AllPPGFSectorTypes}
    operators_f::Vector{Operator}
    operators_i::Vector{Operator}
    propagator::PPGFSector
end

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
    pairs::Vector{InteractionPair{ScalarGF}}
    "List of hybridization function determinants (not implemented yet)"
    determinants::Vector{InteractionDeterminant}
    "List of operator pairs used in accumulation of two-point correlation functions"
    corr_operators::Vector{Tuple{Operator, Operator}}

    "Block matrix representation of the identity operator"
    identity_mat::SectorBlockMatrix
    "Block matrix representation of paired operators (operator_i, operator_f)"
    pair_operator_mat::Vector{Tuple{SectorBlockMatrix, SectorBlockMatrix}}
    "Block matrix representation of corr_operators"
    corr_operators_mat::Vector{Tuple{SectorBlockMatrix, SectorBlockMatrix}}

    """
    Given a subspace index `s_i`, lists indices of all interaction pairs with operator_i
    acting non-trivially on that subspace.
    """
    subspace_attachable_pairs::Vector{Vector{Int64}}

    """
    $(TYPEDSIGNATURES)
    """
    function Expansion(
            ed::ked.EDCore,
            grid::kd.AbstractTimeGrid,
            interaction_pairs::Vector{InteractionPair{ScalarGF}};
            corr_operators::Vector{Tuple{Operator, Operator}} = Tuple{Operator, Operator}[],
            interpolate_ppgf = false) where ScalarGF
        P0 = ppgf.atomic_ppgf(grid, ed)
        dP0 = ppgf.initial_ppgf_derivative(ed, grid.contour.β)
        P = deepcopy(P0)
        P_orders = Vector{typeof(P)}()

        if interpolate_ppgf

            P0_interp = [
                IncSplineImaginaryTimeGF(P0_s, dP0_s) for (P0_s, dP0_s) in zip(P0, dP0)
            ]
            P_interp = [
                IncSplineImaginaryTimeGF(P_s, dP0_s) for (P_s, dP0_s) in zip(P, dP0)
            ]

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

        identity_mat = operator_to_sector_block_matrix(ed, Operator(1.0))
        pair_operator_mat = [
            (operator_to_sector_block_matrix(ed, pair.operator_i),
             operator_to_sector_block_matrix(ed, pair.operator_f))
            for pair in interaction_pairs
        ]
        corr_operators_mat = [
            (operator_to_sector_block_matrix(ed, op1),
             operator_to_sector_block_matrix(ed, op2))
            for (op1, op2) in corr_operators
        ]

        subspace_attachable_pairs = [
            findall(op -> haskey(op[1], s), pair_operator_mat)
            for s in eachindex(ed.subspaces)
        ]

        return new{ScalarGF, typeof(P0)}(
            ed,
            P0,
            P,
            P_orders,
            interaction_pairs,
            [],
            corr_operators,
            identity_mat,
            pair_operator_mat,
            corr_operators_mat,
            subspace_attachable_pairs)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function Expansion(
        hamiltonian::op.OperatorExpr,
        soi::ked.SetOfIndices,
        grid::kd.ImaginaryTimeGrid;
        hybridization::kd.ImaginaryTimeGF{ComplexF64, false},
        nn_interaction::Union{kd.ImaginaryTimeGF{Float64, false}, Nothing} = nothing,
        corr_operators::Vector{Tuple{Operator, Operator}} = Tuple{Operator, Operator}[],
        interpolate_ppgf = false
    )
    @assert kd.norbitals(hybridization) == length(soi)

    #
    # Hybridization
    #

    symmetry_breakers = typeof(hamiltonian)[]

    interactions = InteractionPair{kd.ImaginaryTimeGF{ComplexF64, true}}[]
    for (idx1, n1) in pairs(soi), (idx2, n2) in pairs(soi)
        Δ = kd.ImaginaryTimeGF(
            (t1, t2) -> hybridization[n1, n2, t1, t2],
            hybridization.grid, 1, kd.fermionic, true
        )
        iszero(Δ.mat.data) && continue

        Δ_rev = kd.ImaginaryTimeGF((t1, t2) -> -Δ[t2, t1, false],
                                   Δ.grid, 1, kd.fermionic, true)

        c_dag1 = op.c_dag(idx1...)
        c2 = op.c(idx2...)

        push!(interactions, InteractionPair(c_dag1, c2, Δ))
        push!(interactions, InteractionPair(c2, c_dag1, Δ_rev))

        push!(symmetry_breakers, c_dag1 * c2)
        push!(symmetry_breakers, c2 * c_dag1)
    end

    #
    # NN-interactions
    #

    if !isnothing(nn_interaction)
        @assert kd.norbitals(nn_interaction) == length(soi)

        for (idx1, n1) in pairs(soi), (idx2, n2) in pairs(soi)
            N1 = op.n(idx1...)
            N2 = op.n(idx2...)

            U = kd.ImaginaryTimeGF(
                (t1, t2) -> nn_interaction[n1, n2, t1, t2],
                nn_interaction.grid, 1, kd.fermionic, true
            )

            iszero(U.mat.data) && continue

            push!(interactions, InteractionPair(N1, N2, U))
        end
    end

    ed = ked.EDCore(hamiltonian, soi; symmetry_breakers=symmetry_breakers)

    return Expansion(
        ed,
        grid,
        interactions,
        corr_operators=corr_operators,
        interpolate_ppgf=interpolate_ppgf
    )
end

function set_bold_ppgf!(
        exp::Expansion{ScalarGF, Vector{IncSplineImaginaryTimeGF{ComplexF64, false}}},
        t_i::kd.TimeGridPoint,
        t_f::kd.TimeGridPoint,
        result::SectorBlockMatrix) where ScalarGF <: kd.AbstractTimeGF{ComplexF64, true}
    for (s_i, (s_f, mat)) in result
        @assert s_i == s_f # Boldification must preserve the block structure
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

function set_bold_ppgf_at_order!(
        exp::Expansion{ScalarGF, Vector{IncSplineImaginaryTimeGF{ComplexF64, false}}},
        order::Integer,
        t_i::kd.TimeGridPoint,
        t_f::kd.TimeGridPoint,
        result::SectorBlockMatrix) where ScalarGF <: kd.AbstractTimeGF{ComplexF64, true}
    for (s_i, (s_f, mat)) in result
        @assert s_i == s_f # Boldification must preserve the block structure
        extend!(exp.P_orders[order+1][s_i], mat)
    end
end

function set_bold_ppgf_at_order!(exp::Expansion,
                                 order::Integer,
                                 t_i::kd.TimeGridPoint,
                                 t_f::kd.TimeGridPoint,
                                 result::SectorBlockMatrix)
    for (s_i, (s_f, mat)) in result
        @assert s_i == s_f # Boldification must preserve the block structure
        exp.P_orders[order+1][s_i][t_f, t_i] = mat
    end
end

function add_corr_operators!(exp::Expansion, ops::Tuple{Operator, Operator})
    push!(exp.corr_operators, ops)
    push!(exp.corr_operators_mat,
          (operator_to_sector_block_matrix(exp.ed, ops[1]),
           operator_to_sector_block_matrix(exp.ed, ops[2]))
    )
end

"""
Get all diagrams as combinations of a `Topology` and a list of pseudo particle interaction indices

Parameters
----------

expansion : Pseudo particle expansion
order     : Inch worm perturbation order

Returns
-------

diagrams : Vector with tuples of topologies and pseudo particle interaction indices
"""
function get_diagrams_at_order(
    expansion::Expansion, topologies::Vector{Topology}, order::Int64
    )::Vector{Diagram}

    # Generate all `order` lenght vector of combinations of pseudo particle interaction
    # pair indices
    pair_idx_range = 1:length(expansion.pairs) # range of allowed interaction pair indices
    pair_idxs_combinations = collect(Iterators.product(
        repeat([pair_idx_range], outer=[order])...)
    )

    diagrams = vec([Diagram(topology, pair_idxs) for (topology, pair_idxs) in
            collect(Iterators.product(topologies, pair_idxs_combinations))])

    return diagrams
end

end # module expansion
