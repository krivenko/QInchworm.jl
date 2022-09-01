using Test
using Printf

import LinearAlgebra; trace = LinearAlgebra.tr

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.configuration: Expansion, InteractionPair

import QInchworm.topology_eval: get_topologies_at_order,
                                get_diagrams_at_order

import QInchworm.inchworm: InchwormOrderData,
                           inchworm_step,
                           inchworm_step_bare,
                           inchworm_matsubara!

import QInchworm.ppgf: atomic_ppgf, ImaginaryTimePPGF


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


@testset "project inclusive" begin

    soi_from = KeldyshED.Hilbert.SetOfIndices([[0], [1]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = KeldyshED.Hilbert.SetOfIndices([[0]])
    to = ked.FullHilbertSpace(soi_to)
    
    @test project_inclusive(ked.FockState(0b000), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b001), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b010), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b011), from, to) == 0b001
    
    soi_from = KeldyshED.Hilbert.SetOfIndices([[0], [1]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = KeldyshED.Hilbert.SetOfIndices([[1]])
    to = ked.FullHilbertSpace(soi_to)
    
    @test project_inclusive(ked.FockState(0b000), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b001), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b010), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b011), from, to) == 0b001
    
    soi_from = KeldyshED.Hilbert.SetOfIndices([[0], [1], [2]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = KeldyshED.Hilbert.SetOfIndices([[0]])
    to = ked.FullHilbertSpace(soi_to)
    
    @test project_inclusive(ked.FockState(0b000), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b001), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b010), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b011), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b100), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b101), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b110), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b111), from, to) == 0b001
    
    soi_from = KeldyshED.Hilbert.SetOfIndices([[0], [1], [2]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    to = ked.FullHilbertSpace(soi_to)
    
    @test project_inclusive(ked.FockState(0b001), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b010), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b100), from, to) == 0b010
    @test project_inclusive(ked.FockState(0b110), from, to) == 0b011

end


@testset "reduced density matrix (independent states)" begin

    β = 13.37
    e_vec = [-0.1, 0.0, +0.1]

    H = sum([ e_vec[i] * op.n(i) for i in 1:length(e_vec) ])
    soi = KeldyshED.Hilbert.SetOfIndices([ [i] for i in 1:length(e_vec) ])
    ed = KeldyshED.EDCore(H, soi)

    for i in 1:length(e_vec)
    
        H_small = e_vec[i] * op.n(i)
        soi_small = KeldyshED.Hilbert.SetOfIndices([[i]])
        ed_small = KeldyshED.EDCore(H_small, soi_small)

        ρ_small = density_matrix(ed_small, β)
        ρ_reduced = reduced_density_matrix(ed, ed_small, β)

        @test trace(ρ_reduced) ≈ 1.0
        @test ρ_reduced ≈ ρ_small
    end
    
end


@testset "reduced density matrix (hybridized dimer)" begin

    β = 13.37
    V = 1.0

    H = V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed = KeldyshED.EDCore(H, soi)

    H_small = 0 * op.n(1)
    soi_small = KeldyshED.Hilbert.SetOfIndices([[1]])
    ed_small = KeldyshED.EDCore(H_small, soi_small)

    ρ_small = density_matrix(ed_small, β)
    ρ_reduced = reduced_density_matrix(ed, ed_small, β)

    @test trace(ρ_reduced) ≈ 1.0
    @test ρ_reduced ≈ ρ_small
    
end


@testset "ppgf" begin

    ntau = 5
    V = 1.0
    β = 13.37

    H = V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed = KeldyshED.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)
    
    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    P0 = atomic_ppgf(grid, ed)
    ρ_ppgf = eigenstate_density_matrix(P0)

    #@show ρ
    #@show ρ_ppgf
    @test ρ ≈ ρ_ppgf

    # -- Occupation number density matrices

    ρ_occ = density_matrix(ed, β)
    ρ_occ_ppgf = density_matrix(P0, ed, β)
    #@show ρ_occ
    #@show ρ_occ_ppgf
    @test ρ_occ ≈ ρ_occ_ppgf
    
end


@testset "ppgf assym" begin

    ntau = 5
    V = 1.0
    β = 10.0

    H = 1.0 * op.n(1) + V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed = KeldyshED.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)
    
    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    P0 = atomic_ppgf(grid, ed)
    ρ_ppgf = eigenstate_density_matrix(P0)

    #@show ρ
    #@show ρ_ppgf
    @test ρ ≈ ρ_ppgf

    # -- Occupation number density matrices

    ρ_occ = density_matrix(ed, β)
    ρ_occ_ppgf = density_matrix(P0, ed, β)
    #@show ρ_occ
    #@show ρ_occ_ppgf
    @test ρ_occ ≈ ρ_occ_ppgf

    H_small = 1.0 * op.n(1)
    soi_small = KeldyshED.Hilbert.SetOfIndices([[1]])
    ed_small = KeldyshED.EDCore(H_small, soi_small)
    
    ρ_reduced = reduced_density_matrix(ed, ed_small, β)

    @test trace(ρ_reduced) ≈ 1.0
    #@show ρ_reduced

    
end


function run_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol)

    β = 1.0
    ϵ_1, ϵ_2 = 0.0, 2.0
    V = 0.5

    # -- ED solution

    H_dimer = ϵ_1 * op.n(1) + ϵ_2 * op.n(2) + V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)
    
    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    
    H = ϵ_1 * op.n(1)
    soi = KeldyshED.Hilbert.SetOfIndices([[1]])
    ed = KeldyshED.EDCore(H, soi)

    ρ_ref = Array{ComplexF64}( reduced_density_matrix(ed_dimer, ed, β) )
    
    # -- Hybridization propagator
    
    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ_2, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ_2),
        grid, 1, kd.fermionic, true)

    # -- Pseudo Particle Strong Coupling Expansion

    ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_bwd = InteractionPair(op.c(1), op.c_dag(1), Δ)
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    ρ_0 = density_matrix(expansion.P0, ed, β)

    #@show eigenstate_density_matrix(expansion.P0)
    #@show eigenstate_density_matrix(expansion.P)
    
    #@test ρ_ref ≈ ρ_0
    
    #exit()
    
    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    #@show eigenstate_density_matrix(expansion.P)
    ρ_wrm = density_matrix(expansion.P, ed, β)

    # ρ_ref = 0.4928078037213840 0.5071921962786160 
    #[array([[0.49284566]]), array([[0.50715434]])] cthyb
    
    a = 0.4928077555264891
    b = 0.5071922444735112
    ρ_nca = [[a, 0.], [0., b]]

    @show ρ_0
    @show ρ_ref
    @show ρ_nca
    @show ρ_wrm

    @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
    @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
    @printf "ρ_nca = %16.16f %16.16f \n" ρ_nca[1][1] ρ_nca[2][2]
    @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])
    
    diff_ref = maximum(abs.(ρ_ref - ρ_0))
    diff = maximum(abs.(ρ_ref - ρ_wrm))
    @show diff_ref
    @show diff
    return diff
end


@testset "inchworm_matsubara" begin

    orders = 0:1
    orders_bare = 0:1
    N_chunk = 1000
    max_chunks = 10
    qmc_convergence_atol = 1e-10

    #ntaus = 2 .^ (1:5)
    ntaus = [16*2*2]
    
    @show ntaus

    diffs = [ run_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol) for ntau in ntaus ]

    @show diffs
end
