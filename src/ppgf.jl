# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Authors: Hugo U. R. Strand, Igor Krivenko, Joseph Kleinhenz

"""
Pseudo-particle Green's functions (propagators) of finite fermionic systems and
related tools.

For a system defined by a time-independent Hamiltonian ``\\hat H``, the pseudo-particle
Green's function (PPGF) is

```math
P(z, z') = \\left\\{
\\begin{array}{ll}
-i (-1)^{\\hat N} e^{-i \\hat H(z-z')},& z \\succ -i\\beta \\cap -i\\beta \\succeq z',\\\\
-i e^{-i \\hat H(z-z')},& \\text{otherwise}.
\\end{array}
\\right.
```

In particular, on the imaginary time segment alone one has
``P(\\tau) = -i e^{-\\hat H \\tau}``.

This operator has a block-diagonal structure determined by the symmetry sectors of
``\\hat H``, and is stored as a vector of GF containers corresponding to the individual
diagonal blocks ([`FullTimePPGF`](@ref FullTimePPGF),
[`ImaginaryTimePPGF`](@ref ImaginaryTimePPGF)).

# Exports
$(EXPORTS)
"""
module ppgf

using DocStringExtensions

using LinearAlgebra: Diagonal, tr, I, diagm

using Keldysh; kd = Keldysh;
using KeldyshED; ked = KeldyshED;

using QInchworm.spline_gf: SplineInterpolatedGF

export FullTimePPGF, ImaginaryTimePPGF
export atomic_ppgf
export partition_function, density_matrix
export normalize!

"""
    $(TYPEDSIGNATURES)

Make matrix representation of an operator expression `op_expr` in each invariant subspace
(symmetry sector) defined by the exact diagonalization object `ed`.

**NB!** Requires that the operator expression does not mix symmetry sectors.
"""
function operator_matrix_representation(
    op_expr::ked.OperatorExpr{S}, ed::ked.EDCore)::Vector{Matrix{S}} where {S <: Number}

    op = ked.Operator{ked.FullHilbertSpace, S}(op_expr, ed.full_hs.soi)

    op_sector_matrices = Matrix{S}[]
    for (sidx, subspace) in enumerate(ed.subspaces)
        op_matrix = Matrix{S}(undef, length(subspace), length(subspace))
        i_state = ked.StateVector{ked.HilbertSubspace, S}(subspace)
        for i in eachindex(subspace)
            i_state[i] = one(S)
            f_state = op * i_state
            op_matrix[:, i] = f_state.amplitudes
            i_state[i] = zero(S)
        end
        push!(op_sector_matrices, op_matrix)
    end
    return op_sector_matrices
end

"""
    $(TYPEDSIGNATURES)

Return the total density operator ``\\hat N = \\sum_i n_i``, where ``i`` labels all
single-particle basis states used to construct the exact diagonalization object `ed`.
"""
function total_density_operator(ed::ked.EDCore)
    return sum([ked.Operators.n(label...) for (label, i) in ed.full_hs.soi])
end

# N.B. We cannot use FullTimeGF instead of GenericTimeGF here,
# because FullTimeGF's data storage scheme relies on the symmetry
# properties the pseudo-particle GF's do not possess.

"A single block of an atomic propagator (PPGF) defined on a full Keldysh contour"
const FullTimePPGFSector = kd.GenericTimeGF{ComplexF64, false, kd.FullTimeGrid}
"An atomic propagator (PPGF) defined on a full Keldysh contour"
const FullTimePPGF = Vector{FullTimePPGFSector}
"A single block of an atomic propagator (PPGF) defined on an imaginary time segment"
const ImaginaryTimePPGFSector = kd.ImaginaryTimeGF{ComplexF64, false}
"An atomic propagator (PPGF) defined on an imaginary time segment"
const ImaginaryTimePPGF = Vector{ImaginaryTimePPGFSector}

"""
    $(TYPEDSIGNATURES)

Compute atomic pseudo-particle Green's function on a full contour time `grid` for a
time-independent exact diagonalization problem `ed`.

As the resulting PPGF ``P(z, z')`` is defined up to a multiplier ``e^{-i\\lambda (z-z')}``,
we choose the energy shift ``\\lambda`` to fulfil the normalization property
``i \\mathrm{Tr}[P(-i\\beta, 0)] = 1``.
"""
function atomic_ppgf(grid::kd.FullTimeGrid, ed::ked.EDCore)::FullTimePPGF
    P = [kd.GenericTimeGF(grid, length(s)) for s in ed.subspaces]
    atomic_ppgf!(P, ed)
    return P
end

"""
    $(TYPEDSIGNATURES)

Compute atomic pseudo-particle Green's function on an imaginary time `grid` for a
time-independent exact diagonalization problem `ed`.

As the resulting PPGF ``P(\\tau)`` is defined up to a multiplier ``e^{-\\lambda\\tau}``,
we choose the energy shift ``\\lambda`` to fulfil the normalization property
``i \\mathrm{Tr}[P(\\beta)] = 1``.
"""
function atomic_ppgf(grid::kd.ImaginaryTimeGrid, ed::ked.EDCore)::ImaginaryTimePPGF
    P = [kd.ImaginaryTimeGF(grid, length(s)) for s in ed.subspaces]
    atomic_ppgf!(P, ed)
    return P
end

"""
    $(TYPEDSIGNATURES)

In-place version of [`atomic_ppgf()`](@ref atomic_ppgf) that writes the computed PPGF into
its first argument `P`. If `Δλ` is non-zero, then ``P(z, z')`` is multiplied by
``e^{-i\\Delta\\lambda (z-z')}``.
"""
function atomic_ppgf!(P::Vector, ed::ked.EDCore; Δλ::Float64 = 0.0)
    @assert length(P) == length(ed.subspaces)

    β = first(P).grid.contour.β
    Z = ked.partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[i P(β)] = Tr[ρ] = 1)

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for (P_s, s, E, n) in zip(P, ed.subspaces, ked.energies(ed), N)
        ξ = (-1)^n[1,1] # Statistics sign
        grid = P_s.grid
        _atomic_ppgf_fill_P!(P_s, grid, E, λ + Δλ, ξ)
    end
    return nothing
end

function _atomic_ppgf_fill_P!(P_s::kd.AbstractTimeGF{T, scalar} where {T, scalar},
                              grid::kd.AbstractTimeGrid,
                              E,
                              λ,
                              ξ)
    β = P_s.grid.contour.β
    z_β = grid[kd.imaginary_branch][end]
    Threads.@threads for z1 in grid
	    for z2 in grid[1:z1.cidx]
            Δz = z1.bpoint.val - z2.bpoint.val
            if z1.bpoint.domain == kd.forward_branch &&
                z2.bpoint.domain != kd.forward_branch
                Δz += -im*β
            end
            sign = ξ^(z1.cidx > z_β.cidx && z_β.cidx >= z2.cidx)
            P_s[z1, z2] = -im * sign * Diagonal(exp.(-im * Δz * (E .+ λ)))
	    end
    end
end

function _atomic_ppgf_fill_P!(P_s::kd.ImaginaryTimeGF{T, scalar} where {T, scalar},
                              grid::kd.ImaginaryTimeGrid,
                              E,
                              λ,
                              ξ)
    z2 = grid[kd.imaginary_branch][1]
    z_β = grid[kd.imaginary_branch][end]
    for z1 in grid
        Δz = z1.bpoint.val - z2.bpoint.val
        sign = ξ^(z1.cidx > z_β.cidx && z_β.cidx >= z2.cidx)
        P_s[z1, z2] = -im * sign * Diagonal(exp.(-im * Δz * (E .+ λ)))
    end
end

"""
    $(TYPEDSIGNATURES)

Evaluate a product of vertices at different contour times ``z_n, n=1\\ldots N`` with the
pseudo-particle Green's functions sandwiched in between. The product is padded with the
PPGFs ``P(z_1, z_i)`` and ``P(z_f, z_N)`` at the respective ends of the contour segment
``[z_i, z_f]``.

`vertices` is a contour-time ordered list of triples `(z_n, c_n, o_n)` were:
- `z_n` is a contour time,
- `c_n` is +1/-1 for creation/annihilation operator respectively, and
- `o_n` is a spin-orbital index.

# Parameters
- `ed`:       An object defining the exact diagonalization problem.
- `P`:        The pseudo-particle Green's function as a list of its diagonal blocks.
- `s_i`:      Initial symmetry sector, in which the rightmost PPGF is acting.
- `z_i`:      Initial contour time ``z_i``.
- `z_f`:      Final contour time ``z_f``.
- `vertices`: The list of vertices.

# Returns
The evaluated matrix product and the final symmetry sector, in which the leftmost PPGF is
acting.
"""
function operator_product(ed::ked.EDCore, P, s_i::Integer, z_i, z_f, vertices)

    length(vertices) == 0 && return (P[s_i][z_f, z_i], s_i)

    s_a = s_i
    (z_a, c_a, o_a) = vertices[1]

    prod0 = im * P[s_a][z_a, z_i]
    prod = prod0

    for (vidx, (z_a, c_a, o_a)) in enumerate(vertices)

        connection = c_a > 0 ? ked.cdag_connection : ked.c_connection
        matrix = c_a > 0 ? ked.cdag_matrix : ked.c_matrix

        s_b = connection(ed, o_a, s_a)
        isnothing(s_b) && return zero(prod0), -1

        m_ba = matrix(ed, o_a, s_a)

        if vidx < length(vertices)
            z_b = vertices[vidx + 1][1]
        else
            z_b = z_f
        end

        prod = im * P[s_b][z_b, z_a] * m_ba * prod

        s_a = s_b
    end

    return (prod, s_a)
end

"""
    $(TYPEDSIGNATURES)

Compute the first order pseudo-particle diagram contribution to the single-particle
Green's function ``g_{o_1, o_2}(z, z')`` defined on the full Keldysh contour.

# Parameters
- `P`:  Pseudo-particle Green's function.
- `ed`: An object defining the exact diagonalization problem.
- `o1`: First index of the single-particle Green's function to be computed.
- `o2`: Second index of the single-particle Green's function to be computed.

# Returns
The computed single-particle Green's function.
"""
function first_order_spgf(P::FullTimePPGF,
                          ed::ked.EDCore,
                          o1, o2)::kd.FullTimeGF
    @assert length(P) == length(ed.subspaces)
    g = kd.FullTimeGF(first(P).grid, 1, kd.fermionic, true)
    first_order_spgf!(g, P, ed, o1, o2)
    return g
end

"""
    $(TYPEDSIGNATURES)

Compute the first order pseudo-particle diagram contribution to the single-particle
Green's function ``g_{o_1, o_2}(\\tau)`` defined on the imaginary time segment.

# Parameters
- `P`:  Pseudo-particle Green's function.
- `ed`: An object defining the exact diagonalization problem.
- `o1`: First index of the single-particle Green's function to be computed.
- `o2`: Second index of the single-particle Green's function to be computed.

# Returns
The computed single-particle Green's function.
"""
function first_order_spgf(P::ImaginaryTimePPGF,
                          ed::ked.EDCore,
                          o1, o2)::kd.ImaginaryTimeGF
    @assert length(P) == length(ed.subspaces)
    g = kd.ImaginaryTimeGF(first(P).grid, 1, kd.fermionic, true)
    first_order_spgf!(g, P, ed, o1, o2)
    return g
end

"""
    $(TYPEDSIGNATURES)

In-place version of [`first_order_spgf()`](@ref first_order_spgf) that writes the computed
single-particle Green's function into its first argument `g`.
"""
function first_order_spgf!(g, P, ed::ked.EDCore, o1, o2)
    @assert length(P) == length(ed.subspaces)

    grid = g.grid

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for z1 in grid, z2 in grid

        # Creation/annihilator operator commutation sign
        sign = (-1)^(z1.cidx < z2.cidx)

        # Operator verticies
        v1 = (z1, -1, o1)
        v2 = (z2, +1, o2)

        # (twisted contour) time ordered operator verticies
        v1, v2 = sort([v1, v2], by = x -> x[1].cidx, rev=true)

        # -- Determine start and end time on twisted contour
        if z1.bpoint.domain == kd.imaginary_branch &&
           z2.bpoint.domain == kd.imaginary_branch
            # Equilibrium start at τ = 0 and end at τ = β
            real_time = false
            tau_grid = grid[kd.imaginary_branch]
            z_i, z_f = tau_grid[1], tau_grid[end]
        else
            # Atleast one time is in real-time
            real_time = true

            z_max = sort([z1, z2],
                by = x -> real(x.bpoint.val) - (x.bpoint.domain == kd.imaginary_branch)
            )[end]

            if z_max.bpoint.domain == kd.forward_branch
                z_f = z_max
                z_i = grid[1 + grid[end].cidx - z_max.cidx]
            else
                z_i = z_max
                z_f = grid[1 + grid[end].cidx - z_max.cidx]
            end
        end

        g[z1, z2] = .0
        for (sidx, s) in enumerate(ed.subspaces)
            ξ = (-1)^N[sidx][1, 1]
            prod, sidx_f = operator_product(ed, P, sidx, z_i, z_f, [v2, v1])
            if sidx == sidx_f
                g[z1, z2] += -im * sign * ξ^real_time * tr(prod)
            end
        end
    end
end

"""
    $(TYPEDSIGNATURES)

Check whether a given pseudo-particle Green's function `P` obeys all symmetry relations
between its Keldysh components.

# Parameters
- `P`:  Pseudo-particle Green's function to check.
- `ed`: An object defining the respective exact diagonalization problem.
"""
function check_ppgf_real_time_symmetries(P::FullTimePPGF, ed::ked.EDCore)::Bool
    @assert length(P) == length(ed.subspaces)
    grid = first(P).grid

    grid_bwd = grid[kd.backward_branch]
    zb_i, zb_f = grid_bwd[1], grid_bwd[end]

    grid_fwd = grid[kd.forward_branch]
    zf_i, zf_f = grid_fwd[1], grid_fwd[end]

    # Symmetry between P_{--} and P_{++}

    for zb_1 in grid_bwd
        for zb_2 in grid_bwd[1:zb_1.cidx]
            zb_1.cidx >= zb_2.cidx || return false

            zf_1 = grid[zf_f.cidx - zb_1.cidx + 1]
            zf_2 = grid[zf_f.cidx - zb_2.cidx + 1]

            zb_1.bpoint.val ≈ zf_1.bpoint.val || return false
            zb_2.bpoint.val ≈ zf_2.bpoint.val || return false

            zf_2.cidx >= zf_1.cidx || return false

            for P_s in P
                P_s[zb_1, zb_2] ≈ -conj(P_s[zf_2, zf_1]) || return false
            end
        end
    end

    # Symmetry along anti-diagonal of P_{+-}

    for zf_1 in grid_fwd
        for zb_1 in grid_bwd[1:zf_f.cidx - zf_1.cidx + 1]

            zf_2 = grid[zf_f.cidx - (zb_1.cidx - zb_i.cidx)]
            zb_2 = grid[zb_f.cidx - (zf_1.cidx - zf_i.cidx)]

            zf_1.bpoint.val ≈ zb_2.bpoint.val || return false
            zb_1.bpoint.val ≈ zf_2.bpoint.val || return false

            for P_s in P
                P_s[zf_1, zb_1] ≈ -conj(P_s[zf_2, zb_2]) || return false
            end
        end
    end

    # Symmetry between P_{M-} and P_{+M}

    z_0 = grid[kd.imaginary_branch][1]
    z_β = grid[kd.imaginary_branch][end]
    β = im * z_β.bpoint.val

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for zf in grid_bwd

        zb = grid[zf_f.cidx - zf.cidx + 1]
        zf.bpoint.val ≈ zb.bpoint.val || return false

        for τ in grid[kd.imaginary_branch]
            τ_β = grid[z_β.cidx - (τ.cidx - z_0.cidx)]
            τ_β.bpoint.val ≈ -im*β - τ.bpoint.val || return false

            for (sidx, P_s) in enumerate(P)
                ξ = (-1)^N[sidx][1, 1]
                P_s[τ, zf] ≈ -ξ * conj(P_s[zb, τ_β]) || return false
            end
        end
    end
    return true
end

"""
    $(TYPEDSIGNATURES)

Given a pseudo-particle Green's function `P`, set its values to zero except for
the same-time components ``P(z, z) = -i``.
"""
function set_ppgf_initial_conditions!(P::Union{FullTimePPGF, ImaginaryTimePPGF})
    for P_s in P
        P_s = zero(P_s)
        for z in P_s.grid
            P_s[z, z] += -im * I
        end
    end
end

"""
    $(TYPEDSIGNATURES)

Given a pseudo-particle Green's function `P`, set its values at the real branch edges
``t = 0`` to be consistent with values on the imaginary time branch.

# Parameters
- `P`:  Pseudo-particle Green's function.
- `ed`: Exact diagonalization object used to derive statistical sign pre-factors of PPGF
        sectors.
"""
function ppgf_real_time_initial_conditions!(P::FullTimePPGF, ed::ked.EDCore)
    @assert length(P) == length(ed.subspaces)

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    grid = first(P).grid
    zb0 = grid[kd.backward_branch][end]
    zf0 = grid[kd.forward_branch][1]
    τ_0 = grid[kd.imaginary_branch][1]
    τ_β = grid[kd.imaginary_branch][end]

    for (P_s, n) in zip(P, N)
        for τ in grid[kd.imaginary_branch]
            set_ppgf_symmetric!(P_s, n, τ, zb0, P_s[τ, τ_0])
        end
        P_s[zf0, zb0] = (-1)^n[1, 1] * P_s[τ_β, τ_0]
    end
end

"""
    $(TYPEDSIGNATURES)

Set real-time pseudo-particle Green's function symmetry connected time pairs.

# Parameters
- `P_s`:   Diagonal block of the PPGF to be updated.
- `n`:     The number of particles in the corresponding sector.
- `z1`:    First argument of the PPGF.
- `z2`:    Second argument of the PPGF (must lie on the backward branch).
- `value`: Value to set elements of the PPGF to.
"""
function set_ppgf_symmetric!(P_s::FullTimePPGFSector,
                             n,
                             z1::kd.TimeGridPoint,
                             z2::kd.TimeGridPoint,
                             value)
    grid = P_s.grid

    grid_bwd = grid[kd.backward_branch]
    zb_i, zb_f = grid_bwd[1], grid_bwd[end]
    @assert z2 ∈ grid_bwd

    grid_fwd = grid[kd.forward_branch]
    zf_i, zf_f = grid_fwd[1], grid_fwd[end]

    z_0 = grid[kd.imaginary_branch][1]
    z_β = grid[kd.imaginary_branch][end]

    η = 1

    if z1.bpoint.domain == kd.backward_branch &&
       z2.bpoint.domain == kd.backward_branch
        z3 = grid[zf_f.cidx - z2.cidx + 1]
        z4 = grid[zf_f.cidx - z1.cidx + 1]
    elseif z1.bpoint.domain == kd.imaginary_branch &&
           z2.bpoint.domain == kd.backward_branch
        z3 = grid[zf_f.cidx - z2.cidx + 1]
        z4 = grid[z_β.cidx - (z1.cidx - z_0.cidx)]
        η = -1
    elseif z1.bpoint.domain == kd.forward_branch &&
           z2.bpoint.domain == kd.backward_branch
        z3 = grid[zf_f.cidx - (z2.cidx - zb_i.cidx)]
        z4 = grid[zb_f.cidx - (z1.cidx - zf_i.cidx)]
    else
        @assert false
    end

    ξ = η ^ n[1, 1]
    P_s[z3, z4] = -ξ * conj(value)
    P_s[z1, z2] = value
end

"""
    $(TYPEDSIGNATURES)

Extract the partition function ``Z = i\\mathrm{Tr}[P(-i\\beta, 0)]`` from a un-normalized
pseudo-particle Green's function `P`.
"""
function partition_function(P::Vector{<:kd.AbstractTimeGF})::ComplexF64
    return sum(P, init = 0im) do P_s
        tau_grid = P_s.grid[kd.imaginary_branch]
        τ0, β = tau_grid[1], tau_grid[end]
        im * tr(P_s[β, τ0])
    end
end

"""
    $(TYPEDSIGNATURES)

Normalize a pseudo-particle Green's function `P` by multiplying it by
``e^{-i\\lambda (z-z')}`` with ``\\lambda`` chosen such that
``i\\mathrm{Tr}[P(-i\\beta, 0)] = 1``.

# Returns
The energy shift ``\\lambda``.
"""
function normalize!(P::Vector{<:kd.AbstractTimeGF}, β)
    Z = partition_function(P)
    λ = log(Z) / β
    for P_s in P
        normalize!(P_s, λ)
    end
    return λ
end

"""
    $(TYPEDSIGNATURES)

Multiply a given diagonal block of a pseudo-particle Green's function `P_s` by
``e^{-i\\lambda (z-z')}``.
"""
function normalize!(P_s::kd.AbstractTimeGF, λ)
    tau_grid = P_s.grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    for τ in tau_grid
        P_s[τ, τ_0] = P_s[τ, τ_0] .* exp(-1im * τ.bpoint.val * λ)
    end
end

"""
    $(TYPEDSIGNATURES)

Compute the imaginary time derivative of the atomic pseudo-particle Green's function,
``\\partial_\\tau P(\\tau)`` at ``\\tau=0``.

# Parameters
- `ed`: Exact diagonalization object defining the atomic propagator.
- `β`:  Inverse temperature.

# Returns
Value of the derivative as a block-diagonal matrix (a list of blocks).
"""
function initial_ppgf_derivative(ed::ked.EDCore, β::Float64)
    Z = sum([sum(exp.(-β * eig.eigenvalues)) for eig in ed.eigensystems])
    λ = log(Z) / β

    dP = []
    for eig in ed.eigensystems
        dP_s = -im * diagm(-eig.eigenvalues .- λ)
        push!(dP, dP_s)
    end
    return dP
end

"""
    $(TYPEDSIGNATURES)

Extract the equilibrium density matrix ``\\rho = i P(-i\\beta, 0)`` from a normalized
pseudo-particle Green's function `P`. The density matrix is block-diagonal and is returned
as a vector of blocks.
"""
function density_matrix(P::Vector{<:kd.AbstractTimeGF})
    @assert !isempty(P)
    τ_0, τ_β = kd.branch_bounds(first(P).grid, kd.imaginary_branch)
    return [1im * P_s[τ_β, τ_0] for P_s in P]
end

"""
    $(TYPEDSIGNATURES)

Take a partial trace of a pseudo-particle Green's function.

# Parameters
- `P`:          Pseudo-particle Green's function.
- `ed`:         Exact diagonalization object compatible with `P`.
- `target_soi`: A subset of creation/annihilation operator labels defining the target
                subsystem, in which the reduced PPGF acts.

# Returns
The reduced PPGF written in the Fock state basis of the target subsystem.
"""
function reduced_ppgf(P::Vector{<:kd.AbstractTimeGF},
                      ed::ked.EDCore,
                      target_soi::ked.SetOfIndices)
    return ked.partial_trace(ked.tofockbasis(P, ed), ed, target_soi)
end

end # module ppgf
