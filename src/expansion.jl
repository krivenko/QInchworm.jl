# QInchworm.jl
#
# Copyright (C) 2021-2024 I. Krivenko, H. U. R. Strand and J. Kleinhenz
#
# QInchworm.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# QInchworm.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Igor Krivenko, Hugo U. R. Strand

"""
Strong coupling pseudo-particle expansion problem.

# Exports
$(EXPORTS)
"""
module expansion

using DocStringExtensions

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.sector_block_matrix: SectorBlockMatrix, operator_to_sector_block_matrix
using QInchworm.ppgf
using QInchworm.diagrammatics: Topology
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.spline_gf: IncSplineImaginaryTimeGF, extend!

export Expansion, InteractionPair, add_corr_operators!

const Operator = op.RealOperatorExpr

"Supported container types for a single block of an atomic propagator (PPGF)"
const AllPPGFSectorTypes = Union{
    ppgf.FullTimePPGFSector,
    ppgf.ImaginaryTimePPGFSector,
    SplineInterpolatedGF{ppgf.FullTimePPGFSector, ComplexF64, false},
    SplineInterpolatedGF{ppgf.ImaginaryTimePPGFSector, ComplexF64, false},
    IncSplineImaginaryTimeGF{ComplexF64, false}
}

"Supported container types for a full atomic propagator (PPGF)"
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

Data type for pseudo-particle interactions containing two operators and one scalar
propagator.

Indexed access to the operators stored in a `pair::InteractionPair` is supported:
`pair[1]` and `pair[2]` are equivalent to `pair.operator_i` and `pair.operator_f`
respectively.

# Fields
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

The `Expansion` structure contains the components needed to define a strong coupling
pseudo-particle expansion problem.

# Fields
$(FIELDS)
"""
struct Expansion{ScalarGF <: kd.AbstractTimeGF{ComplexF64, true}, PPGF <: AllPPGFTypes}
    "Exact diagonalization solver for the local degrees of freedom"
    ed::ked.EDCore
    "Non-interacting propagator (pseudo-particle Green's function)"
    P0::PPGF
    "Interacting propagator (pseudo-particle Green's function)"
    P::PPGF
    "List of pseudo-particle interactions"
    pairs::Vector{InteractionPair{ScalarGF}}
    "List of hybridization function determinants (not implemented yet)"
    determinants::Vector{InteractionDeterminant}
    "List of operator pairs used in accumulation of two-point correlation functions"
    corr_operators::Vector{Tuple{Operator, Operator}}

    "Block matrix representation of the identity operator"
    identity_mat::SectorBlockMatrix
    "Block matrix representation of paired operators (`operator_i`, `operator_f`)"
    pair_operator_mat::Vector{Tuple{SectorBlockMatrix, SectorBlockMatrix}}
    "Block matrix representation of `corr_operators`"
    corr_operators_mat::Vector{Tuple{SectorBlockMatrix, SectorBlockMatrix}}

    """
    Given a subspace index `s_i`, lists indices of all interaction pairs with operator_i
    acting non-trivially on that subspace.
    """
    subspace_attachable_pairs::Vector{Vector{Int64}}

    @doc """
        $(TYPEDSIGNATURES)

    # Parameters
    - `ed`:                Exact diagonalization solution of the local problem.
    - `grid`:              Contour time grid to define the local propagators on.
    - `interaction_pairs`: The list of pair interactions to expand in.
    - `corr_operators`:    The list of operator pairs used in accumulation of two-point
                           correlation functions.
    - `interpolate_ppgf`:  Use a quadratic spline interpolation to represent and evaluate
                           the local propagators. *Currently works only with the imaginary
                           time propagators.*
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

A higher-level constructor of [`Expansion`](@ref Expansion) that solves the local problem
defined by a Hamiltonian and internally generates a list of pseudo-particle pair
interactions from hybridization and ``nn``-interaction functions.

# Parameters
- `hamiltonian`:      Hamiltonian of the local problem.
- `soi`:              An ordered set of indices carried by creation/annihilation operators
                      of the local problem.
- `grid`:             Imaginary time grid to define the local propagators on.
- `hybridization`:    A matrix-valued hybridization function ``\\Delta_{ij}(\\tau)``. A
                      correspondence between the matrix elements ``(i, j)`` and operators
                      ``c^\\dagger_i, c_j`` is established by `soi`.
- `nn_interaction`:   A matrix-valued ``nn``-interaction function ``U_{ij}(\\tau)``.
                      A correspondence between the matrix elements ``(i, j)`` and operators
                      ``n_i, n_j`` is established by `soi`.
- `corr_operators`:   The list of operator pairs used in accumulation of two-point
                      correlation functions.
- `interpolate_ppgf`: Use a quadratic spline interpolation to represent and evaluate
                      the local propagators.
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

"""
    $(TYPEDSIGNATURES)

Add a pair of operators ``(A, B)`` used to measure the two-point correlator
``\\langle A(t_1) B(t_2)\\rangle`` to `expansion`.

# Parameters
- `expansion`: Pseudo-particle expansion.
- `ops`:       The pair of operators ``(A, B)``.
"""
function add_corr_operators!(expansion::Expansion, ops::Tuple{Operator, Operator})
    push!(expansion.corr_operators, ops)
    push!(expansion.corr_operators_mat,
          (operator_to_sector_block_matrix(expansion.ed, ops[1]),
           operator_to_sector_block_matrix(expansion.ed, ops[2]))
    )
    return nothing
end

end # module expansion
