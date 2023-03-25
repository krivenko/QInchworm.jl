using Keldysh; kd = Keldysh

using Interpolations: interpolate, scale, BSpline, Cubic, OnGrid

using QInchworm.utility: NeumannBC
using QInchworm.utility: IncrementalSpline, extend!

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
