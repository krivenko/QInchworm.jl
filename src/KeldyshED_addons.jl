module KeldyshED_addons

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf: ImaginaryTimePPGF

""" Projects a FockState form one FullHilbertSpace to another FullHilbertSpace.

The projection is done "inclusive". Parts of the input state `fs` that are in the `to` FullHilbertSpace are retained while any parts of the state `fs` not included in the `to` target space are discarded. 

"""

function project_inclusive(fs::ked.FockState, from::ked.FullHilbertSpace, to::ked.FullHilbertSpace)
    @assert fs ∈ from
    #@show fs
    common_keys = intersect(keys(from.soi), keys(to.soi))
    #@show common_keys
    @assert length(common_keys) > 0
    out = ked.Hilbert.FockState(0)
    for key in common_keys
        #@show key
        from_idx = from.soi[key]
        to_idx = to.soi[key]
        #@show from_idx
        #@show to_idx
        #@show (0b1 << (from_idx-1))
        #@show (fs & (0b1 << from_idx-1))
        #@show (0b1 << (to_idx-1))
        if fs & (0b1 << (from_idx-1)) > 0
            out += (0b1 << (to_idx-1))
        end
        #@show out
    end
    @assert out ∈ to
    #@show out
    return out
end

"""Project a state from one Hilbert space to another Hilbert space/subspace"""
function project_inclusive(
    sv::ked.StateVector{HSType, S},
    target_space::TargetHSType,
    from::ked.FullHilbertSpace, to::ked.FullHilbertSpace
    ) where {HSType, S, TargetHSType}

    proj_sv = ked.StateVector{TargetHSType, S}(target_space)
    for (i, a) in pairs(sv.amplitudes)
        f = sv.hs[i]
        #@show f
        f_p = project_inclusive(f, from, to)
        #@show f_p
        if f_p in target_space
            proj_sv[ked.getstateindex(target_space, f_p)] += a
        end
    end
    proj_sv
end

"""Project a state from one Hilbert space to another Hilbert space/subspace"""
function project_trace(
    f_diff::ked.FockState,
    sv::ked.StateVector{HSType, S},
    from::ked.FullHilbertSpace, to::ked.FullHilbertSpace
    ) where {HSType, S}

    diff_soi = ked.Hilbert.SetOfIndices(
        setdiff(keys(from.soi), keys(to.soi)) )
    diff_hs = ked.FullHilbertSpace(diff_soi)

    #@show from
    #@show to
    #@show diff_hs
    
    proj_sv = ked.StateVector{ked.FullHilbertSpace, S}(to)
    for (i, a) in pairs(sv.amplitudes)
        f = sv.hs[i]
        #println("-----------")
        #@show f

        #@show f_diff
        f_diff_proj = project_inclusive(f, from, diff_hs)
        #@show f_diff_proj
        
        if f_diff_proj == f_diff
            f_p = project_inclusive(f, from, to)
            #@show f_p
            if f_p in to
                proj_sv[ked.getstateindex(to, f_p)] += a
            end
        end
    end
    proj_sv
end


function reduced_density_matrix(
    ρ, ed::ked.EDCore, target_ed::ked.EDCore, β::Float64)

    n = length(target_ed.full_hs)
    ρ_out = zeros(Float64, n, n)

    diff_soi = ked.Hilbert.SetOfIndices(
        setdiff(keys(ed.full_hs.soi), keys(target_ed.full_hs.soi)) )
    diff_hs = ked.FullHilbertSpace(diff_soi)

    for (ρss, hss, eig) in zip(ρ, ed.subspaces, ed.eigensystems)

        for i in 1:length(hss)
            ρ_ii = ρss[i, i]
            vec = eig.unitary_matrix[:, i]
            state = ked.StateVector(hss, vec)

            for j in 1:length(diff_hs)
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
    return reduced_density_matrix(ρ, ed, target_ed, β)
end


function density_matrix(ρ, ed::ked.EDCore, β::Float64)

    n = length(ed.full_hs)
    ρ_out = zeros(Float64, n, n)

    for (ρss, hss, eig) in zip(ρ, ed.subspaces, ed.eigensystems)

        for i in 1:length(hss)
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
    ρ = KeldyshED.density_matrix(ed, β)
    return density_matrix(ρ, ed, β)
end


function eigenstate_density_matrix(P::ImaginaryTimePPGF)
    grid = P[1].grid
    τ_0, τ_β = grid[1], grid[end]
    [ 1im * P_s[τ_β, τ_0] for P_s in P ]
end


function density_matrix(P::ImaginaryTimePPGF, ed::ked.EDCore, β::Float64)
    ρ = eigenstate_density_matrix(P)
    return density_matrix(ρ, ed, β)
end

end # module KeldyshED_addons
