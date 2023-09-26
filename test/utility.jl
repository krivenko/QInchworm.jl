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
   A1, A2, A3, A4, A5 = [rand(rng, dims[i + 1], dims[i]) for i = 1:5]

   lmp = LazyMatrixProduct(Float64, 6)

   pushfirst!(lmp, A1)
   @test eval!(lmp) == A1

   pushfirst!(lmp, A2)
   pushfirst!(lmp, A3)
   pushfirst!(lmp, A4)
   pushfirst!(lmp, A5)

   @test eval!(lmp) ≈ A5 * A4 * A3 * A2 * A1
   @test eval!(lmp) ≈ A5 * A4 * A3 * A2 * A1

   popfirst!(lmp, 2)
   @test eval!(lmp) ≈ A3 * A2 * A1

   popfirst!(lmp)
   @test eval!(lmp) ≈ A2 * A1

   pushfirst!(lmp, A3)
   pushfirst!(lmp, A4)
   @test eval!(lmp) ≈ A4 * A3 * A2 * A1

   popfirst!(lmp, 3)
   @test eval!(lmp) == A1
   @test eval!(lmp) == A1
end
