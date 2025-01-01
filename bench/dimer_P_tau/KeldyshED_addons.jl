# QInchworm.jl
#
# Copyright (C) 2021-2025 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Authors: Hugo U. R. Strand, Igor Krivenko

module KeldyshED_addons

using LinearAlgebra: tr, diagm

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED

using QInchworm.expansion: AllPPGFTypes
using QInchworm.ppgf

""" Projects a FockState from one FullHilbertSpace to another FullHilbertSpace.

The projection is done "inclusive". Parts of the input state `fs` that are in the `to`
FullHilbertSpace are retained while any parts of the state `fs` not included in the `to`
target space are discarded.
"""
function project_inclusive(fs::ked.Hilbert.FockState,
                           from::ked.FullHilbertSpace,
                           to::ked.FullHilbertSpace)
    @boundscheck fs ∈ from
    common_keys = intersect(keys(from.soi), keys(to.soi))
    @boundscheck length(common_keys) > 0
    out = ked.Hilbert.FockState(0)
    for key in common_keys
        from_idx = from.soi[key]
        to_idx = to.soi[key]
        if fs & (0b1 << (from_idx-1)) > 0
            out += (0b1 << (to_idx-1))
        end
    end
    @boundscheck out ∈ to
    return out
end

"""Project a state from one Hilbert space to another Hilbert space/subspace"""
function project_inclusive(
    sv::ked.StateVector{HSType, S},
    target_space::TargetHSType,
    from::ked.FullHilbertSpace,
    to::ked.FullHilbertSpace
    ) where {HSType, S, TargetHSType}

    proj_sv = ked.StateVector{TargetHSType, S}(target_space)
    for (i, a) in pairs(sv.amplitudes)
        f = sv.hs[i]
        f_p = project_inclusive(f, from, to)
        if f_p in target_space
            proj_sv[ked.getstateindex(target_space, f_p)] += a
        end
    end
    return proj_sv
end

"""Project a state from one Hilbert space to another Hilbert space/subspace"""
function project_trace(
    f_diff::ked.Hilbert.FockState,
    sv::ked.StateVector{HSType, S},
    from::ked.FullHilbertSpace, to::ked.FullHilbertSpace
    ) where {HSType, S}

    diff_soi = ked.Hilbert.SetOfIndices(
        setdiff(keys(from.soi), keys(to.soi)) )
    diff_hs = ked.FullHilbertSpace(diff_soi)

    proj_sv = ked.StateVector{ked.FullHilbertSpace, S}(to)
    for (i, a) in pairs(sv.amplitudes)
        f = sv.hs[i]
        f_diff_proj = project_inclusive(f, from, diff_hs)

        if f_diff_proj == f_diff
            f_p = project_inclusive(f, from, to)
            if f_p in to
                proj_sv[ked.getstateindex(to, f_p)] += a
            end
        end
    end
    return proj_sv
end

function reduced_density_matrix(
    ρ, ed::ked.EDCore, target_ed::ked.EDCore)

    n = length(target_ed.full_hs)
    ρ_out = zeros(Float64, n, n)

    diff_soi = ked.Hilbert.SetOfIndices(
        setdiff(keys(ed.full_hs.soi), keys(target_ed.full_hs.soi)) )
    diff_hs = ked.FullHilbertSpace(diff_soi)

    for (ρss, hss, eig) in zip(ρ, ed.subspaces, ed.eigensystems)

        for i in eachindex(hss)
            ρ_ii = ρss[i, i]
            vec = eig.unitary_matrix[:, i]
            state = ked.StateVector(hss, vec)

            for j in eachindex(diff_hs)
                diff_f = diff_hs[j]
                p_state = project_trace(
                    diff_f, state, ed.full_hs, target_ed.full_hs)
                vec = p_state.amplitudes
                ρ_out += ρ_ii * (vec * vec')
            end
        end
    end
    return ρ_out
end


function reduced_density_matrix(
    ed::ked.EDCore, target_ed::ked.EDCore, β::Float64)
    ρ = ked.density_matrix(ed, β)
    return reduced_density_matrix(ρ, ed, target_ed)
end

function reduced_ppgf(P::T, ed::ked.EDCore, target_ed::ked.EDCore) where {T <: AllPPGFTypes}
    grid = P[1].grid
    n = length(target_ed.full_hs)
    m = length(ed.full_hs)
    P_red = kd.ImaginaryTimeGF(grid, n)

    τ_0 = grid[1]
    for τ_i in grid
        ρ = [ P_s[τ_i, τ_0] for P_s in P ]
        ρ_red = reduced_density_matrix(ρ, ed, target_ed)
        P_red[τ_i, τ_0] = ρ_red
    end

    τ_β = grid[end]
    Z = im * tr(P_red[τ_β, τ_0])
    Z = Z / (m - n)
    λ = log(Z) / grid.contour.β
    ppgf.normalize!(P_red, λ)
    Z = im * tr(P_red[τ_β, τ_0])

    for τ_i in grid
        P_red[τ_i, τ_0] = P_red[τ_i, τ_0] / (m - n)
    end

    return P_red
end

function occupation_number_basis_ppgf(P::AllPPGFTypes, ed::ked.EDCore)
    grid = P[1].grid
    n = length(ed.full_hs)
    P_out = kd.ImaginaryTimeGF(grid, n)

    τ_0 = grid[1]
    for τ_i in grid
        ρ = [ P_s[τ_i, τ_0] for P_s in P ]
        P_out[τ_i, τ_0] = density_matrix(ρ, ed)
    end

    return P_out
end

function density_matrix(ρ, ed::ked.EDCore)
    n = length(ed.full_hs)
    ρ_out = zeros(ComplexF64, n, n)

    for (ρss, hss, eig) in zip(ρ, ed.subspaces, ed.eigensystems)
        for i in eachindex(hss)
            ρ_ii = ρss[i, i]
            vec = eig.unitary_matrix[:, i]
            state = ked.StateVector(hss, vec)
            p_state = ked.project(state, ed.full_hs)
            vec = p_state.amplitudes

            ρ_out += ρ_ii * (vec * vec')
        end
    end
    return ρ_out
end

function density_matrix(ed::ked.EDCore, β::Float64)
    ρ = ked.density_matrix(ed, β)
    return density_matrix(ρ, ed)
end

function eigenstate_density_matrix(P::AllPPGFTypes)
    grid = P[1].grid
    τ_0, τ_β = grid[1], grid[end]
    ρ = [1im * P_s[τ_β, τ_0] for P_s in P]
    return ρ
end

function density_matrix(P::AllPPGFTypes, ed::ked.EDCore)
    ρ = eigenstate_density_matrix(P)
    return density_matrix(ρ, ed)
end

# KeldyshED routines specialized to 1D ImaginaryTimeGrid

function evolution_operator_imtime(ed::EDCore,
                                   grid::ImaginaryTimeGrid)::
                                   Vector{ImaginaryTimeGF{ComplexF64, false}}
    [ImaginaryTimeGF(grid, length(es.eigenvalues)) do t1, t2
        diagm(exp.(-im * es.eigenvalues * (t1.bpoint.val - t2.bpoint.val)))
    end
    for es in ed.eigensystems]
end

function tofockbasis_imtime(S::Vector{GF}, ed::EDCore) where {GF <: ImaginaryTimeGF}
    S_fock = GF[]
    for (s, es) in zip(S, ed.eigensystems)
        U = es.unitary_matrix
        push!(S_fock, similar(s))
        t2 = S_fock[end].grid[1]
        for t1 in S_fock[end].grid
            S_fock[end][t1, t2] = U * s[t1, t2] * adjoint(U)
        end
    end
    return S_fock
end

function partial_trace_imtime(S::Vector{GF},
                              ed::EDCore,
                              target_soi::ked.SetOfIndices) where {GF <: ImaginaryTimeGF}
    target_hs = ked.FullHilbertSpace(target_soi)
    fmap = ked.factorized_basis_map(ed.full_hs, target_hs)

    S_target = similar(S[1], length(target_hs))

    for (Sss, hss) in zip(S, ed.subspaces)
        for (i1, fs1) in pairs(hss), (i2, fs2) in pairs(hss)
            i_target1, i_env1 = fmap[ked.getstateindex(ed.full_hs, fs1)]
            i_target2, i_env2 = fmap[ked.getstateindex(ed.full_hs, fs2)]

            i_env1 != i_env2 && continue

            t2 = S_target.grid[1]
            for t1 in S_target.grid
                S_target[i_target1, i_target2, t1, t2] += Sss[i1, i2, t1, t2]
            end
        end
    end

    return S_target
end

function reduced_evolution_operator_imtime(ed::EDCore,
                                    target_soi::ked.SetOfIndices,
                                    grid::ImaginaryTimeGrid)
    S = tofockbasis_imtime(evolution_operator_imtime(ed, grid), ed)
    return partial_trace_imtime(S, ed, target_soi)
end

end # module KeldyshED_addons
