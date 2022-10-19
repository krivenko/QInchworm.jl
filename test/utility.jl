using Test

import Keldysh; kd = Keldysh

using Interpolations: interpolate, scale, BSpline, Cubic, OnGrid

import QInchworm.utility: get_ref
import QInchworm.utility: NeumannBC
import QInchworm.utility: IncrementalSpline, extend!

@testset "get_ref()" begin
    tmax = 1.
    β = 5.

    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));

    for ref in [0.0, 0.5, 2.0, 5.0, 5.5, 6.5]
        @test get_ref(contour, contour(ref)) == ref
    end
end

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

"""
@testset "IncrementalSpline" begin
    f(x) = exp(2*x)

    knots = LinRange(3, 4, 100)
    data = f.(knots)
    der1 = 2*data[1]

    spline = IncrementalSpline(knots, data[1], der1)
    for x in data[2:end]
        extend!(spline, x)
    end

    knots_fine = LinRange(3, 4, 1000)
    @test isapprox(spline.(knots_fine), f.(knots_fine), rtol=1e-7)
end
"""