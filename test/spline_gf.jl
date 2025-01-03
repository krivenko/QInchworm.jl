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
# Author: Igor Krivenko

using Test

using Keldysh; kd = Keldysh

using QInchworm.spline_gf: SplineInterpolatedGF,
                           update_interpolant!,
                           update_interpolants!

@testset "spline_gf" begin

    β = 10.
    nτ = 6
    ϵ = +0.1 # Energy level

    dos = kd.DeltaDOS(ϵ)

    @testset "Imaginary time GF" begin
        contour = kd.ImaginaryContour(β=β);
        grid = kd.ImaginaryTimeGrid(contour, nτ);

        @testset "scalar = true" begin
            G = kd.ImaginaryTimeGF(dos, grid)
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
            G = kd.ImaginaryTimeGF(grid, 2) do t1, t2
                ones(2, 2) * kd.dos2gf(dos, β, t1.bpoint, t2.bpoint)
            end

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
