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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/.
#
# Author: Igor Krivenko

using Test

using Random: AbstractRNG
import Random


using QInchworm.ScrambledSobol: ScrambledSobolSeq, next!, skip!

@testset "ScrambledSobolSeq" begin

    @testset "Unscrambled" begin

        @testset "D=0" begin
            s = ScrambledSobolSeq(0)

            @test ndims(s) == 0
            @test "$(s)" == "0-dimensional scrambled Sobol sequence on [0,1]^0"
        end

        @testset "D=1" begin
            s = ScrambledSobolSeq(1)

            @test ndims(s) == 1
            @test "$(s)" == "1-dimensional scrambled Sobol sequence on [0,1]^1"

            # Reference sequence from SciPy
            # scipy.stats.qmc.Sobol(d=1, scramble=False, bits=32).random_base2(m=3)
            ref = [[0.],
                   [0.5],
                   [0.75],
                   [0.25],
                   [0.375],
                   [0.875],
                   [0.625],
                   [0.125]]

            @test [next!(s) for _ in 1:8] == ref

            s = ScrambledSobolSeq(1)
            skip!(s, 3, exact=true)
            @test [next!(s) for _ in 1:5] == ref[4:end]

            s = ScrambledSobolSeq(1)
            skip!(s, 3) # Skips 4 points
            @test [next!(s) for _ in 1:4] == ref[5:end]
        end

        @testset "D=5" begin
            s = ScrambledSobolSeq(5)

            @test ndims(s) == 5
            @test "$(s)" == "5-dimensional scrambled Sobol sequence on [0,1]^5"

            # Reference sequence from SciPy
            # scipy.stats.qmc.Sobol(d=5, scramble=False, bits=32).random_base2(m=3)
            ref = [[0.   , 0.   , 0.   , 0.   , 0.   ],
                   [0.5  , 0.5  , 0.5  , 0.5  , 0.5  ],
                   [0.75 , 0.25 , 0.25 , 0.25 , 0.75 ],
                   [0.25 , 0.75 , 0.75 , 0.75 , 0.25 ],
                   [0.375, 0.375, 0.625, 0.875, 0.375],
                   [0.875, 0.875, 0.125, 0.375, 0.875],
                   [0.625, 0.125, 0.875, 0.625, 0.625],
                   [0.125, 0.625, 0.375, 0.125, 0.125]]

            @test [next!(s) for _ in 1:8] == ref

            s = ScrambledSobolSeq(5)
            skip!(s, 3, exact=true)
            @test [next!(s) for _ in 1:5] == ref[4:end]

            s = ScrambledSobolSeq(5)
            skip!(s, 3) # Skips 4 points
            @test [next!(s) for _ in 1:4] == ref[5:end]
        end
    end

    # A mock random number generator
    struct MockRNG <: AbstractRNG end

    # This function is called by `ScrambledSobolSeq`'s constructor when an instance of
    # `MockRNG` is passed to it. This way generation of the scrambling parameters becomes
    # reproducible.
    function Random.rand(rng::MockRNG, S::Vector, dims::Integer...)
        size = prod(dims)
        return reshape([S[count_ones(i) % length(S) + 1] for i in 0:(size - 1)], dims...)
    end

    # The reference sequences are produced by SciPy with a dynamically patched function
    # `scipy._lib._util.rng_integers`:
    #
    # def mock_rng_integers(rng, high, /, size, dtype):
    #     r = list(range(high))
    #     seq = np.array([r[i.bit_count() % len(r)] for i in range(np.prod(size))],
    #                    dtype=dtype)
    #     return np.ascontiguousarray(seq.reshape(size, order='F'))
    #
    # import scipy._lib._util
    # scipy._lib._util.rng_integers = mock_rng_integers

    @testset "Scrambled" begin

        @testset "D=0" begin
            s = ScrambledSobolSeq(0, scramble_rng = MockRNG())

            @test ndims(s) == 0
            @test "$(s)" == "0-dimensional scrambled Sobol sequence on [0,1]^0"
        end

        @testset "D=1" begin
            s = ScrambledSobolSeq(1, scramble_rng = MockRNG())

            @test ndims(s) == 1
            @test "$(s)" == "1-dimensional scrambled Sobol sequence on [0,1]^1"

            # Reference sequence from SciPy
            # scipy.stats.qmc.Sobol(d=1, scramble=False, bits=32).random_base2(m=3)
            ref = [[0.58754596626386],
                   [0.49999999976717],
                   [0.16245403350331],
                   [0.75            ],
                   [0.96254596626386],
                   [0.12499999976717],
                   [0.28745403350331],
                   [0.625           ]]

            @test isapprox([next!(s) for _ in 1:8], ref, atol=1e-10)

            s = ScrambledSobolSeq(1, scramble_rng = MockRNG())
            skip!(s, 3, exact=true)
            @test isapprox([next!(s) for _ in 1:5], ref[4:end], atol=1e-10)

            s = ScrambledSobolSeq(1, scramble_rng = MockRNG())
            skip!(s, 3) # Skips 4 points
            @test isapprox([next!(s) for _ in 1:4], ref[5:end], atol=1e-10)
        end

        @testset "D=5" begin
            s = ScrambledSobolSeq(5, scramble_rng = MockRNG())

            @test ndims(s) == 5
            @test "$(s)" == "5-dimensional scrambled Sobol sequence on [0,1]^5"

            # Reference sequence from SciPy
            # scipy.stats.qmc.Sobol(d=5, scramble=False, bits=32).random_base2(m=3)
            ref = [[0.69067839533091, 0.39461669255979, 0.79655789281242, 0.74460224201903,
                    0.02035492961295],
                   [0.21048400853761, 0.85785256954841, 0.00012016296387, 0.35785256954841,
                    0.71060703252442],
                   [0.44202061719261, 0.10309065226465, 0.51562499976717, 0.84527815226465,
                    0.77645712764934],
                   [0.95352603914216, 0.6444934909232,  0.28154231142253, 0.23984332964756,
                    0.46133853914216],
                   [0.81713465577923, 0.14444692479447, 0.17158174957149, 0.09527359250933,
                    0.27652141125873],
                   [0.32857844652608, 0.60308610741049, 0.87500189244747, 0.98977093980648,
                    0.96126889111474],
                   [0.06579133123159, 0.35784567031078, 0.39050294435583, 0.49453348806128,
                    0.52041653101332],
                   [0.58553650532849, 0.89456789125688, 0.65656804572791, 0.60784568521194,
                    0.21053553675301]]

            @test isapprox([next!(s) for _ in 1:8], ref, atol=1e-10)

            s = ScrambledSobolSeq(5, scramble_rng = MockRNG())
            skip!(s, 3, exact=true)
            @test isapprox([next!(s) for _ in 1:5], ref[4:end], atol=1e-10)

            s = ScrambledSobolSeq(5, scramble_rng = MockRNG())
            skip!(s, 3) # Skips 4 points
            @test isapprox([next!(s) for _ in 1:4], ref[5:end], atol=1e-10)
        end
    end
end
