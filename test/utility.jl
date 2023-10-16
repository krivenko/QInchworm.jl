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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Author: Igor Krivenko

using Test

using Keldysh; kd = Keldysh

using Interpolations: interpolate, scale, BSpline, Cubic, OnGrid
using Random: MersenneTwister

using QInchworm.utility: NeumannBC
using QInchworm.utility: IncrementalSpline, extend!
using QInchworm.utility: LazyMatrixProduct, eval!

@testset "NeumannBC" begin
    f(x) = exp(2*x)

    knots = LinRange(3, 4, 100)
    data = f.(knots)
    left_derivative = 2*exp(2*3)
    right_derivative = 2*exp(2*4)

    bc = NeumannBC(OnGrid(), left_derivative, right_derivative)

    spline = scale(interpolate(data, BSpline(Cubic(bc))), knots)

    knots_fine = LinRange(3, 4, 1000)
    @test isapprox(spline.(knots_fine), f.(knots_fine), rtol=1e-8)
end

@testset "IncrementalSpline" begin
    f(x) = exp(2*x) + im * sin(x)

    knots = LinRange(3, 4, 100)
    data = f.(knots)
    der1 = 2*exp(2*knots[1]) + im * cos(knots[1])

    spline = IncrementalSpline(knots, data[1], der1)
    for x in data[2:end]
        extend!(spline, x)
    end

    knots_fine = LinRange(3, 4, 1000)
    @test isapprox(spline.(knots_fine), f.(knots_fine), rtol=1e-7)
end

@testset "LazyMatrixProduct" begin
   rng = MersenneTwister(123456)
   dims = [rand(rng, 1:8) for i = 1:6]
   A = [rand(rng, dims[i + 1], dims[i]) for i = 1:5]

   lmp = LazyMatrixProduct(Float64, 6)

   pushfirst!(lmp, A[1])
   @test eval!(lmp) == A[1]

   pushfirst!(lmp, A[2])
   pushfirst!(lmp, A[3])
   pushfirst!(lmp, A[4])
   pushfirst!(lmp, A[5])

   @test eval!(lmp) ≈ A[5] * A[4] * A[3] * A[2] * A[1]
   @test eval!(lmp) ≈ A[5] * A[4] * A[3] * A[2] * A[1]

   popfirst!(lmp, 2)
   @test eval!(lmp) ≈ A[3] * A[2] * A[1]

   popfirst!(lmp)
   @test eval!(lmp) ≈ A[2] * A[1]

   pushfirst!(lmp, A[3])
   pushfirst!(lmp, A[4])
   @test eval!(lmp) ≈ A[4] * A[3] * A[2] * A[1]

   popfirst!(lmp, 3)
   @test eval!(lmp) == A[1]
   @test eval!(lmp) == A[1]
end
