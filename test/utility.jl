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
