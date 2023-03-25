using Keldysh; kd = Keldysh

using QInchworm.spline_gf: SplineInterpolatedGF,
                           update_interpolant!,
                           update_interpolants!

@testset "spline_gf" begin

    β = 10.
    ntau = 6
    ϵ = +0.1 # Energy level

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

            @testset "Interpolation on a reduced τ-segment" begin
                  G_int_red = SplineInterpolatedGF(deepcopy(G), τ_max=grid[3])
                  t_pts = [(t1, t2) for t1=grid, t2=grid if 0 <= (t1.cidx - t2.cidx) <= 2]

                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  # setindex!()
                  G_int_red[1, 1, grid[4], grid[2], τ_max=grid[3]] = 0.5im
                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  # update_interpolant!()
                  G_int_red.GF[1, 1, grid[4], grid[3]] = 0.6im
                  update_interpolant!(G_int_red, 1, 1, τ_max=grid[3])
                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  # update_interpolants!()
                  G_int_red.GF[grid[2], grid[1]] = 0.7im
                  update_interpolants!(G_int_red, τ_max=grid[3])
                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  update_interpolants!(G_int_red, τ_max=grid[2])
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[2].bpoint)
            end

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

            @testset "Interpolation on a reduced τ-segment" begin
                  G_int_red = SplineInterpolatedGF(deepcopy(G), τ_max=grid[3])
                  t_pts = [(t1, t2) for t1=grid, t2=grid if 0 <= (t1.cidx - t2.cidx) <= 2]

                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  # setindex!()
                  G_int_red[2, 2, grid[4], grid[2], τ_max=grid[3]] = 0.5im
                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  # update_interpolant!()
                  G_int_red.GF[2, 2, grid[4], grid[3]] = 0.6im
                  update_interpolant!(G_int_red, 2, 2, τ_max=grid[3])
                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  # update_interpolants!()
                  G_int_red.GF[grid[2], grid[1]] = [2.0im 1.0im; 4.0im 3.0im]
                  update_interpolants!(G_int_red, τ_max=grid[3])
                  @test [G_int_red.GF(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts] ≈
                        [G_int_red(t1.bpoint, t2.bpoint) for (t1, t2) in t_pts]
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[1].bpoint)

                  update_interpolants!(G_int_red, τ_max=grid[2])
                  @test_throws BoundsError G_int_red(grid[4].bpoint, grid[2].bpoint)
            end
        end
    end
end
