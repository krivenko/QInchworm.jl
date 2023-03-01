using Test
using Printf

using LinearAlgebra: tr, diagm

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: atomic_ppgf, ImaginaryTimePPGF

using QInchworm.KeldyshED_addons: project_inclusive,
                                  project_trace,
                                  reduced_density_matrix,
                                  density_matrix,
                                  eigenstate_density_matrix

@testset "project inclusive" begin

    soi_from = ked.Hilbert.SetOfIndices([[0], [1]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = ked.Hilbert.SetOfIndices([[0]])
    to = ked.FullHilbertSpace(soi_to)

    @test project_inclusive(ked.FockState(0b000), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b001), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b010), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b011), from, to) == 0b001

    soi_from = ked.Hilbert.SetOfIndices([[0], [1]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = ked.Hilbert.SetOfIndices([[1]])
    to = ked.FullHilbertSpace(soi_to)

    @test project_inclusive(ked.FockState(0b000), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b001), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b010), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b011), from, to) == 0b001

    soi_from = ked.Hilbert.SetOfIndices([[0], [1], [2]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = ked.Hilbert.SetOfIndices([[0]])
    to = ked.FullHilbertSpace(soi_to)

    @test project_inclusive(ked.FockState(0b000), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b001), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b010), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b011), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b100), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b101), from, to) == 0b001
    @test project_inclusive(ked.FockState(0b110), from, to) == 0b000
    @test project_inclusive(ked.FockState(0b111), from, to) == 0b001

    soi_from = ked.Hilbert.SetOfIndices([[0], [1], [2]])
    from = ked.FullHilbertSpace(soi_from)
    soi_to = ked.Hilbert.SetOfIndices([[1], [2]])
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
    soi = ked.Hilbert.SetOfIndices([ [i] for i in 1:length(e_vec) ])
    ed = ked.EDCore(H, soi)

    for i in 1:length(e_vec)

        H_small = e_vec[i] * op.n(i)
        soi_small = ked.Hilbert.SetOfIndices([[i]])
        ed_small = ked.EDCore(H_small, soi_small)

        ρ_small = density_matrix(ed_small, β)
        ρ_reduced = reduced_density_matrix(ed, ed_small, β)

        @test tr(ρ_reduced) ≈ 1.0
        @test ρ_reduced ≈ ρ_small
    end

end


@testset "reduced density matrix (hybridized dimer)" begin

    β = 13.37
    V = 1.0

    H = V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H, soi)

    H_small = 0 * op.n(1)
    soi_small = ked.Hilbert.SetOfIndices([[1]])
    ed_small = ked.EDCore(H_small, soi_small)

    ρ_small = density_matrix(ed_small, β)
    ρ_reduced = reduced_density_matrix(ed, ed_small, β)

    @test tr(ρ_reduced) ≈ 1.0
    @test ρ_reduced ≈ ρ_small

end


@testset "ppgf" begin

    ntau = 5
    V = 1.0
    β = 13.37

    H = V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    P0 = atomic_ppgf(grid, ed)
    ρ_ppgf = eigenstate_density_matrix(P0)
    @test ρ ≈ ρ_ppgf

    # -- Occupation number density matrices

    ρ_occ = density_matrix(ed, β)
    ρ_occ_ppgf = density_matrix(P0, ed)
    @test ρ_occ ≈ ρ_occ_ppgf

end


@testset "ppgf assym" begin
    ntau = 5
    V = 1.0
    β = 10.0

    H = 1.0 * op.n(1) + V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    P0 = atomic_ppgf(grid, ed)
    ρ_ppgf = eigenstate_density_matrix(P0)
    @test ρ ≈ ρ_ppgf

    # -- Occupation number density matrices

    ρ_occ = density_matrix(ed, β)
    ρ_occ_ppgf = density_matrix(P0, ed)
    @test ρ_occ ≈ ρ_occ_ppgf

    H_small = 1.0 * op.n(1)
    soi_small = ked.Hilbert.SetOfIndices([[1]])
    ed_small = ked.EDCore(H_small, soi_small)

    ρ_reduced = reduced_density_matrix(ed, ed_small, β)

    @test tr(ρ_reduced) ≈ 1.0
end


function get_reduced_density_matrix_hubbard_dimer(β, U, ϵ_1, ϵ_2, V_1, V_2)
    H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))

    H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) +
        V_1 * ( op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1) ) +
        V_2 * ( op.c_dag(2) * op.c(4) + op.c_dag(4) * op.c(2) )

    soi_dimer = ked.Hilbert.SetOfIndices([[1], [2], [3], [4]])
    ed_dimer = ked.EDCore(H_dimer, soi_dimer)

    soi_small = ked.Hilbert.SetOfIndices([[1], [2]])
    ed_small = ked.EDCore(H_imp, soi_small)

    ρ_reduced = reduced_density_matrix(ed_dimer, ed_small, β)

    return ρ_reduced
end


@testset "reduced density matrix (hybridized hubbard dimer)" begin
    β = 1.0
    U = 0.0
    ϵ_1, ϵ_2 = 0.0, 2.0

    #@printf "===============================================================\n"
    V_1, V_2 = 0.5, 0.0
    ρ_1 = get_reduced_density_matrix_hubbard_dimer(β, U, ϵ_1, ϵ_2, V_1, V_2)

    #@printf "===============================================================\n"
    V_1, V_2 = 0.0, 0.5
    ρ_2 = get_reduced_density_matrix_hubbard_dimer(β, U, ϵ_1, ϵ_2, V_1, V_2)

    # Permutation matrix for switching |1> and |2>
    P = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

    ρ_1 = P * ρ_1 * P # Permute states in ρ_1 so that it becomes identical to ρ_2

    @test all(ρ_1 - diagm(diag(ρ_1)) .≈ 0.)
    @test all(ρ_2 - diagm(diag(ρ_2)) .≈ 0.)

    @test tr(ρ_2) ≈ 1.0
    @test tr(ρ_1) ≈ 1.0

    @test ρ_1 ≈ ρ_2
end
