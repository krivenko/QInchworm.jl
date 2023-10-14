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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/.
#
# Authors: Hugo U. R. Strand, Igor Krivenko, Joseph Kleinhenz

module ppgf

using LinearAlgebra: Diagonal, tr, I, diagm

using Keldysh; kd = Keldysh;
using KeldyshED; ked = KeldyshED;
import KeldyshED: partition_function

using QInchworm.spline_gf: SplineInterpolatedGF

export FullTimePPGF, ImaginaryTimePPGF
export atomic_ppgf
export operator_product
export partition_function, density_matrix
export normalize!

"""
Get matrix representation of operator expression in each sector

NB! Requires that the operator expression does not mix symmetry sectors
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

function total_density_operator(ed::ked.EDCore)
    return sum([ked.Operators.n(label...) for (label, i) in ed.full_hs.soi])
end

# N.B. We cannot use FullTimeGF instead of GenericTimeGF here,
# because FullTimeGF's data storage scheme relies on the symmetry
# properties the pseudo-particle GF's do not possess.
const FullTimePPGFSector = kd.GenericTimeGF{ComplexF64, false, kd.FullTimeGrid}
const FullTimePPGF = Vector{FullTimePPGFSector}
const ImaginaryTimePPGFSector = kd.ImaginaryTimeGF{ComplexF64, false}
const ImaginaryTimePPGF = Vector{ImaginaryTimePPGFSector}

const AllImaginaryTimeGF = Union{
    kd.ImaginaryTimeGF{ComplexF64, false},
    SplineInterpolatedGF{kd.ImaginaryTimeGF{ComplexF64, false}, ComplexF64, false}
}

"""
Compute atomic pseudo-particle Green's function on the time grid
for a time-independent problem defined by the EDCore instance.
"""
function atomic_ppgf(grid::kd.FullTimeGrid, ed::ked.EDCore)::FullTimePPGF
    P = [kd.GenericTimeGF(grid, length(s)) for s in ed.subspaces]
    atomic_ppgf!(P, ed)
    return P
end

function atomic_ppgf(grid::kd.ImaginaryTimeGrid, ed::ked.EDCore)::ImaginaryTimePPGF
    P = [kd.ImaginaryTimeGF(grid, length(s)) for s in ed.subspaces]
    atomic_ppgf!(P, ed)
    return P
end

function atomic_ppgf!(P::Vector, ed::ked.EDCore; Δλ::Float64 = 0.0)
    @assert length(P) == length(ed.subspaces)

    β = first(P).grid.contour.β
    Z = partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[P0(β)]=Tr[ρ]=1)

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for (P_s, s, E, n) in zip(P, ed.subspaces, ked.energies(ed), N)
        ξ = (-1)^n[1,1] # Statistics sign
        grid = P_s.grid
        atomic_ppgf_fill_P!(P_s, grid, E, λ + Δλ, ξ)
    end
end

function atomic_ppgf_fill_P!(P_s::kd.AbstractTimeGF{T, scalar} where {T, scalar},
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

function atomic_ppgf_fill_P!(P_s::kd.ImaginaryTimeGF{T, scalar} where {T, scalar},
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
    operator_product(...)

Evaluate a product of vertices at different contour times `z_i` with
the pseudo-particle Green's function sandwiched in between.

`vertices` is a contour-time ordered list of triples `(z_i, c_i, o_i)` were:
  `z_i` is a contour time,
  `c_i` is +1/-1 for creation/annihilation operator respectively, and
  `o_i` is a spin-orbital index
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
Compute the first order pseudo-particle diagram contribution to
the single-particle Green's function g_{o1, o2}(z, z')
"""
function first_order_spgf(P::FullTimePPGF,
                          ed::ked.EDCore,
                          o1, o2)::kd.FullTimeGF
    @assert length(P) == length(ed.subspaces)
    g = kd.FullTimeGF(first(P).grid, 1, kd.fermionic, true)
    first_order_spgf!(g, P, ed, o1, o2)
    return g
end

function first_order_spgf(P::ImaginaryTimePPGF,
                          ed::ked.EDCore,
                          o1, o2)::kd.ImaginaryTimeGF
    @assert length(P) == length(ed.subspaces)
    g = kd.ImaginaryTimeGF(first(P).grid, 1, kd.fermionic, true)
    first_order_spgf!(g, P, ed, o1, o2)
    return g
end

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

function check_ppgf_real_time_symmetries(P::FullTimePPGF, ed)
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

function set_ppgf_initial_conditions!(P::Union{FullTimePPGF, ImaginaryTimePPGF})
    for P_s in P
        P_s = zero(P_s)
        for z in P_s.grid
            P_s[z, z] += -im * I
        end
    end
end

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
        P_s[zf0, zb0] = (-1)^n[1,1] * P_s[τ_β, τ_0]
    end
end

"""
Set real-time ppgf symmetry connected time pairs

NB! times has to be in the inching region with z2 ∈ backward_branch.
"""
function set_ppgf_symmetric!(P_s::FullTimePPGF, n, z1, z2, val)
    grid = P_s.grid

    grid_bwd = grid[kd.backward_branch]
    zb_i, zb_f = grid_bwd[1], grid_bwd[end]

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

    ξ = η^n[1, 1]
    P_s[z3, z4] = -ξ * conj(val)
    P_s[z1, z2] = val
end

function partition_function(P::Vector{<:kd.AbstractTimeGF})::ComplexF64
    return sum(P, init = 0im) do P_s
        tau_grid = P_s.grid[kd.imaginary_branch]
        τ0, β = tau_grid[1], tau_grid[end]
        im * tr(P_s[β, τ0])
    end
end

"""Set all time translation invariant values of the Matsubara branch"""
function set_matsubara!(g::kd.GenericTimeGF{T, scalar, kd.FullTimeGrid} where {T, scalar},
                        τ,
                        value)
    tau_grid = g.grid[kd.imaginary_branch]

    τ_0 = tau_grid[1]
    τ_beta = tau_grid[end]

    sidx = τ.cidx
    eidx = τ_beta.cidx

    for τ_1 in g.grid[sidx:eidx]
        i1 = τ_1.cidx
        i2 = τ_0.cidx + τ_1.cidx - τ.cidx
        t1 = g.grid[i1]
        t2 = g.grid[i2]
        g[t1, t2] = value
    end
end

function set_matsubara!(g::AllImaginaryTimeGF, τ, value)
    tau_grid = g.grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    g[τ, τ_0] = value
end

function normalize!(P::Vector{<:kd.AbstractTimeGF}, β)
    Z = partition_function(P)
    λ = log(Z) / β
    for P_s in P
        normalize!(P_s, λ)
    end
    return λ
end

function normalize!(P_s::kd.AbstractTimeGF, λ)
    tau_grid = P_s.grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    for τ in tau_grid
        P_s[τ, τ_0] = P_s[τ, τ_0] .* exp(-1im * τ.bpoint.val * λ)
    end
end

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

function density_matrix(P::Vector{<:kd.AbstractTimeGF})
    @assert !isempty(P)
    τ_0, τ_β = kd.branch_bounds(first(P).grid, kd.imaginary_branch)
    return [1im * P_s[τ_β, τ_0] for P_s in P]
end

function reduced_ppgf(P::Vector{<:kd.AbstractTimeGF},
                      ed::ked.EDCore,
                      target_soi::ked.SetOfIndices)
    return ked.partial_trace(ked.tofockbasis(P, ed), ed, target_soi)
end

end # module ppgf
