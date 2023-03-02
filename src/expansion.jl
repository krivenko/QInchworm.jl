module expansion

using DocStringExtensions

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm: SectorBlockMatrix
using QInchworm.ppgf
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

    """
    $(TYPEDSIGNATURES)
    """
    function Expansion(ed::ked.EDCore,
                       grid::kd.AbstractTimeGrid,
                       interaction_pairs::Vector{InteractionPair{ScalarGF}};
                       corr_operators::Vector{Tuple{Operator, Operator}} = Tuple{Operator, Operator}[],
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

        return new{ScalarGF, typeof(P0)}(ed, P0, P, P_orders, interaction_pairs, [], corr_operators)
    end
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
    sbm
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

# TODO: These should probably be removed
function set_bold_ppgf!(P::Vector{IncSplineImaginaryTimeGF{ComplexF64, false}},
    t_i::kd.TimeGridPoint,
    t_f::kd.TimeGridPoint,
    result::SectorBlockMatrix)
    for (s_i, (s_f, mat)) in result
        # Boldification must preserve the block structure
        @assert s_i == s_f
        extend!(P[s_i], mat)
    end
end

end # module expansion