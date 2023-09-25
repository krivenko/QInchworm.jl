using Test

using MPI; MPI.Init()
using HDF5

using LinearAlgebra: diag

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.expansion: Expansion, InteractionPair, get_diagrams_at_order
using QInchworm.diagrammatics: get_topologies_at_order

using QInchworm.ppgf
using QInchworm.spline_gf: SplineInterpolatedGF
using QInchworm.inchworm: inchworm!
using QInchworm.mpi: ismaster

@testset "dimer" begin

    function run_dimer(nτ, orders, orders_bare, N_samples; interpolate_gfs=false)

        β = 1.0
        ϵ_1, ϵ_2 = 0.5, 2.0
        V = 0.5

        # -- ED solution

        H_dimer = ϵ_1 * op.n(1) + ϵ_2 * op.n(2) +
                  V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
        soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
        ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)

        # -- Impurity problem

        contour = kd.ImaginaryContour(β=β);
        grid = kd.ImaginaryTimeGrid(contour, nτ);

        H = ϵ_1 * op.n(1)
        soi = KeldyshED.Hilbert.SetOfIndices([[1]])
        ed = KeldyshED.EDCore(H, soi)

        ρ_ref = reduced_density_matrix(ed_dimer, soi, β)

        P0_dimer = ppgf.atomic_ppgf(grid, ed_dimer)
        P_red = ppgf.reduced_ppgf(P0_dimer, ed_dimer, soi)

        # -- Hybridization propagator

        Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ_2), grid)
        Δ_rev = kd.ImaginaryTimeGF((t1, t2) -> -Δ[t2, t1, false],
                                   grid, 1, kd.fermionic, true)

        # -- Pseudo Particle Strong Coupling Expansion

        if interpolate_gfs
            ip_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
            ip_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(Δ_rev))
            expansion = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=true)
            println("Using spline GFS")
        else
            ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
            ip_bwd = InteractionPair(op.c(1), op.c_dag(1), Δ_rev)
            expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])
        end

        ρ_0 = full_hs_matrix(ppgf.density_matrix(expansion.P0), ed)

        inchworm!(expansion, grid, orders, orders_bare, N_samples)

        if interpolate_gfs
            P = [ p.GF for p in expansion.P ]
            ppgf.normalize!(P, β)
            ρ_wrm = full_hs_matrix(ppgf.density_matrix(P), ed)
        else
            ppgf.normalize!(expansion.P, β)
            ρ_wrm = full_hs_matrix(ppgf.density_matrix(expansion.P), ed)
        end

        if ismaster()
            @show real(diag(ρ_0))
            @show real(diag(ρ_ref))
            @show real(diag(ρ_wrm))
        end

        return maximum(abs.(ρ_ref - ρ_wrm))
    end

    @testset "order1" begin
        nτ = 32
        orders = 0:1
        N_samples = 8 * 2^4

        diff_interp = run_dimer(nτ, orders, orders, N_samples, interpolate_gfs=true)
        @show diff_interp
        @test diff_interp < 1e-4

        diff_linear = run_dimer(nτ, orders, orders, N_samples, interpolate_gfs=false)
        @show diff_linear
        @test diff_linear < 1e-4
    end

    @testset "order3" begin
        nτ = 32
        orders = 0:3
        N_samples = 8 * 2^4

        diff_interp = run_dimer(nτ, orders, orders, N_samples, interpolate_gfs=true)
        @show diff_interp
        @test diff_interp < 1e-4

        diff_linear = run_dimer(nτ, orders, orders, N_samples, interpolate_gfs=false)
        @show diff_linear
        @test diff_linear < 1e-4
    end
end

@testset "hubbard_dimer" begin

    function run_hubbard_dimer(ntau, orders, orders_bare, N_samples)

        β = 1.0
        U = 4.0
        ϵ_1, ϵ_2 = 0.0 - 0.5*U, 0.0
        V_1 = 0.5
        V_2 = 0.5

        # -- ED solution

        H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))

        H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) +
            V_1 * ( op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1) ) +
            V_2 * ( op.c_dag(2) * op.c(4) + op.c_dag(4) * op.c(2) )

        soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2], [3], [4]])
        ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)

        # -- Impurity problem

        contour = kd.ImaginaryContour(β=β);
        grid = kd.ImaginaryTimeGrid(contour, ntau);

        soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
        ed = KeldyshED.EDCore(H_imp, soi)

        ρ_ref = reduced_density_matrix(ed_dimer, soi, β)

        # -- Hybridization propagator

        Δ_1 = V_1^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ_2), grid)
        Δ_2 = V_2^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ_2), grid)
        Δ_1_rev = kd.ImaginaryTimeGF((t1, t2) -> -Δ_1[t2, t1, false],
                                     grid, 1, kd.fermionic, true)
        Δ_2_rev = kd.ImaginaryTimeGF((t1, t2) -> -Δ_2[t2, t1, false],
                                     grid, 1, kd.fermionic, true)

        # -- Pseudo Particle Strong Coupling Expansion

        ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ_1)
        ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), Δ_1_rev)
        ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ_2)
        ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), Δ_2_rev)
        expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

        ρ_0 = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P0), ed), ed)

        inchworm!(expansion, grid, orders, orders_bare, N_samples)

        ppgf.normalize!(expansion.P, β)
        ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P), ed), ed)

        if ismaster()
            @show real(diag(ρ_0))
            @show real(diag(ρ_ref))
            @show real(diag(ρ_wrm))
        end

        return maximum(abs.(ρ_ref - ρ_wrm))
    end

    @testset "order1" begin
        nτ = 32
        orders = 0:1
        N_samples = 8 * 2^5

        diff = run_hubbard_dimer(nτ, orders, orders, N_samples)
        @show diff
        @test diff < 1e-4
    end

    @testset "order2" begin
        nτ = 32
        orders = 0:2
        N_samples = 8 * 2^5

        diff = run_hubbard_dimer(nτ, orders, orders, N_samples)
        @show diff
        @test diff < 1e-4
    end

end
