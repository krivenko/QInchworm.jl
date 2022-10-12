using Test

import Keldysh; kd = Keldysh

import QInchworm.spline_gf: SplineInterpolatedGF,
                            update_interpolant!,
                            update_interpolants!

β = 10.
ntau = 6
ϵ = +0.1 # Energy level

@testset "spline_gf" begin

    @testset "Imaginary time GF" begin
        contour = kd.ImaginaryContour(β=β);
        grid = kd.ImaginaryTimeGrid(contour, ntau);

        G_func = (t1, t2) -> -1.0im *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ)

        @testset "scalar = true" begin
            G = kd.ImaginaryTimeGF(G_func, grid, 1, kd.fermionic, true)
            G_int = SplineInterpolatedGF(deepcopy(G))

            @test eltype(G_int) == ComplexF64
            @test kd.norbitals(G_int) == 1
            @test G_int.grid == G.grid

            # Check that G and G_int match on the grid
            @test [G(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            # getindex()
            @test G[1, 1, grid[3], grid[4]] == G_int[1, 1, grid[3], grid[4]]
            @test G[grid[3], grid[4]] == G_int[grid[3], grid[4]]

            # setindex!()
            G[1, 1, grid[3], grid[4]] = 0.5im
            G_int[1, 1, grid[3], grid[4]] = 0.5im
            @test G[grid[3], grid[4]] == G_int[grid[3], grid[4]]

            # Check that G and G_int still match after the setindex!() call
            @test [G(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            G[grid[2], grid[1]] = 1.0im
            G_int[grid[2], grid[1]] = 1.0im
            @test G[grid[2], grid[1]] == G_int[grid[2], grid[1]]

            # Check that G and G_int still match after the setindex!() call
            @test [G(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            # update_interpolant!()
            G_int.GF[1, 1, grid[3], grid[4]] = 0.6im
            update_interpolant!(G_int, 1, 1)
            @test [G_int.GF(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            # update_interpolants!()
            G_int.GF[grid[2], grid[1]] = 0.7im
            update_interpolants!(G_int)
            @test [G_int.GF(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]
        end

        @testset "scalar = false" begin
            G = kd.ImaginaryTimeGF((t1, t2) -> ones(2, 2) * G_func(t1, t2),
                                   grid, 2, kd.fermionic, false)
            G_int = SplineInterpolatedGF(deepcopy(G))

            @test eltype(G_int) == ComplexF64
            @test kd.norbitals(G_int) == 2
            @test G_int.grid == G.grid

            # Check that G and G_int match on the grid
            @test [G(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            # getindex()
            @test G[2, 2, grid[3], grid[4]] == G_int[2, 2, grid[3], grid[4]]
            @test G[grid[3], grid[4]] == G_int[grid[3], grid[4]]

            # setindex!()
            G[2, 2, grid[3], grid[4]] = 0.5im
            G_int[2, 2, grid[3], grid[4]] = 0.5im
            @test G[2, 2, grid[3], grid[4]] == G_int[2, 2, grid[3], grid[4]]

            # Check that G and G_int still match after the setindex!() call
            @test [G(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            G[grid[2], grid[1]] = [1.0im 2.0im; 3.0im 4.0im]
            G_int[grid[2], grid[1]] = [1.0im 2.0im; 3.0im 4.0im]
            @test G[grid[2], grid[1]] == G_int[grid[2], grid[1]]

            # Check that G and G_int still match after the setindex!() call
            @test [G(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            # update_interpolant!()
            G_int.GF[2, 2, grid[3], grid[4]] = 0.6im
            update_interpolant!(G_int, 2, 2)
            @test [G_int.GF(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]

            # update_interpolants!()
            G_int.GF[grid[2], grid[1]] = [2.0im 1.0im; 4.0im 3.0im]
            update_interpolants!(G_int)
            @test [G_int.GF(t1.bpoint, t2.bpoint) for t1=grid, t2=grid] ≈
                  [G_int(t1.bpoint, t2.bpoint) for t1=grid, t2=grid]
        end
    end
end