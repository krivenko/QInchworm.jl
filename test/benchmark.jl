using Test
using Printf

import LinearAlgebra; trace = LinearAlgebra.tr

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf: atomic_ppgf, ImaginaryTimePPGF

import QInchworm.KeldyshED_addons:
    project_inclusive, project_trace, reduced_density_matrix, density_matrix, eigenstate_density_matrix

#import QInchworm.configuration: Expansion, InteractionPair

#import QInchworm.topology_eval: get_topologies_at_order,
#                                get_diagrams_at_order

#import QInchworm.inchworm: InchwormOrderData,
#                           inchworm_step,
#                           inchworm_step_bare,
#                           inchworm_matsubara!

#import QInchworm.ppgf: atomic_ppgf, ImaginaryTimePPGF



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
