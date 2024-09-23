# QInchworm.jl
#
# Copyright (C) 2021-2024 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
using StableRNGs: StableRNG

using QInchworm.utility: NeumannBC
using QInchworm.utility: IncrementalSpline, extend!
using QInchworm.utility: LazyMatrixProduct, eval!
using QInchworm.utility: RandomSeq

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

   lmp = LazyMatrixProduct(Float64, 6, 8, dims[1])

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

@testset "RandomSeq" begin

    @testset "D=0" begin
        s = RandomSeq{StableRNG}(0)
        @test ndims(s) == 0
    end

    @testset "D=1" begin
        s = RandomSeq{StableRNG}(1)
        @test ndims(s) == 1

        # Reference sequence from StableRNGs:
        # rand(StableRNG(0), Float64, 8)
        ref = [[0.19506488073747286],
               [0.025779311523037363],
               [0.9055427466847816],
               [0.7734523504688744],
               [0.8843643358740858],
               [0.07887971712448505],
               [0.569894630482412],
               [0.1859290813601362]
        ]

        @test [next!(s) for _ in 1:8] ≈ ref

        s = RandomSeq{StableRNG}(1)
        skip!(s, 3, exact=true)
        @test [next!(s) for _ in 1:5] ≈ ref[4:end]

        s = RandomSeq{StableRNG}(1)
        skip!(s, 3) # Skips 4 points
        @test [next!(s) for _ in 1:4] ≈ ref[5:end]
    end

    @testset "D=5" begin
        s = RandomSeq{StableRNG}(5)
        @test ndims(s) == 5

        # Reference sequence from StableRNGs
        # rng = StableRNG(0)
        # [rand(rng, Float64, 5) for _ in 1:8]
        ref = [[0.19506488073747286, 0.025779311523037363, 0.9055427466847816,
                0.7734523504688744, 0.8843643358740858],
               [0.07887971712448505, 0.569894630482412, 0.1859290813601362,
                0.3502633405739193, 0.07892729440270685],
               [0.31458032385783596, 0.1485569608098214, 0.2545598088968246,
                0.04835716319427674, 0.007041346527125292],
               [0.05084835898020357, 0.8724974721884622, 0.16051021802467536,
                0.5840743840364566, 0.4382718269681136],
               [0.9528405584114081, 0.23290440928617473, 0.585915354467456,
                0.8064312206579658, 0.8042024139742923],
               [0.15152973035530826, 0.126977517436196, 0.269545129918906,
                0.31763240936542725, 0.6241531180514104],
               [0.13501424175448973, 0.7762634763959406, 0.7198777098828519,
                0.016887314264266262, 0.16689840464120498],
               [0.5297498177820077, 0.7676637636059831, 0.46539490738355216,
                0.09040271797573474, 0.6757828873970544]
        ]

        @test [next!(s) for _ in 1:8] ≈ ref

        s = RandomSeq{StableRNG}(5)
        skip!(s, 3, exact=true)
        @test [next!(s) for _ in 1:5] ≈ ref[4:end]

        s = RandomSeq{StableRNG}(5)
        skip!(s, 3) # Skips 4 points
        @test [next!(s) for _ in 1:4] ≈ ref[5:end]
    end
end
