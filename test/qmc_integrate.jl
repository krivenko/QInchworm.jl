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
# Authors: Igor Krivenko, Hugo U. R. Strand

using Test

using Keldysh; kd = Keldysh

using QInchworm.qmc_integrate: contour_integral,
                               contour_integral_n_samples,
                               make_trans_f,
                               make_jacobian_f,
                               ExpModelFunctionTransform,
                               RootTransform,
                               SortTransform,
                               DoubleSimplexRootTransform

@testset verbose=true "qmc_integrate" begin

    tmax = 1.
    β = 5.

    # Real-time Kadanoff-Baym contour
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β))

    @testset "ExpModelFunctionTransform" begin

        τ = 5.0

        @testset "d = 1, constant integrand" begin
            let d = 1,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? 1.0 : .0),
                N = 2^17
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(0.5), τ),
                                       N=N)
                @test isapprox(val, -0.4, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(2.0), τ),
                                       N=N)
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(5.0), τ),
                                       N=N)
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.5), τ),
                                       N=N)
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.9), τ),
                                       N=N)
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? t[1].val : .0im),
                N = 2^17
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(0.5), τ),
                                       N=N)
                @test isapprox(val, -0.28, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(2.0), τ),
                                       N=N)
                @test isapprox(val, -0.905, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(5.0), τ),
                                       N=N)
                @test isapprox(val, -8.405, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.5), τ),
                                       N=N)
                @test isapprox(val, -12.78, rtol=1e-4)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.9), τ),
                                       N=N)
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? 1.0 : .0),
                N = 2^18
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(0.5), τ),
                                       N=N)
                @test isapprox(val, 0.08, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(2.0), τ),
                                       N=N)
                @test isapprox(val, -0.095+0.9im, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(5.0), τ),
                                       N=N)
                @test isapprox(val, -7.595+3.6im, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.5), τ),
                                       N=N)
                @test isapprox(val, -12.42+2.0im, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.9), τ),
                                       N=N)
                @test isapprox(val, -12.5, rtol=5e-3)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? t[1].val * t[2].val : .0im),
                N = 2^19
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(0.5), τ),
                                       N=N)
                @test isapprox(val, 0.0392, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(2.0), τ),
                                       N=N)
                @test isapprox(val, 0.409513, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(5.0), τ),
                                       N=N)
                @test isapprox(val, 35.322, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.5), τ),
                                       N=N)
                @test isapprox(val, 81.6642, rtol=5e-3)
                val = contour_integral(f, c,
                                       ExpModelFunctionTransform(d, c, c(6.9), τ),
                                       N=N)
                @test isapprox(val, 78.125, rtol=5e-3)
            end
        end
    end

    @testset "ExpModelFunctionTransform (N_samples)" begin

        τ = 5.0

        @testset "d = 1, constant integrand" begin
            let d = 1,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? 1.0 : .0)
                N_samples = 2^16
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(0.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, -0.4, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(2.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(5.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.9), τ),
                    N_samples=N_samples)
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? t[1].val : .0im),
                N_samples = 2^15
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(0.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, -0.28, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(2.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, -0.905, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(5.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, -8.405, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, -12.78, rtol=1e-4)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.9), τ),
                    N_samples=N_samples)
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? 1.0 : .0),
                N_samples = 2^15
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(0.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, 0.08, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(2.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, -0.095+0.9im, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(5.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, -7.595+3.6im, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, -12.42+2.0im, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.9), τ),
                    N_samples=N_samples)
                @test isapprox(val, -12.5, rtol=5e-3)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2,
                c = contour,
                t_i = c(0.1),
                f = t -> (kd.heaviside(c, t[end], t_i) ? t[1].val * t[2].val : .0im),
                N_samples = 2^15
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(0.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, 0.0392, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(2.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, 0.409513, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(5.0), τ),
                    N_samples=N_samples)
                @test isapprox(val, 35.322, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.5), τ),
                    N_samples=N_samples)
                @test isapprox(val, 81.6642, rtol=5e-3)
                val, N = contour_integral_n_samples(f, c,
                    ExpModelFunctionTransform(d, c, c(6.9), τ),
                    N_samples=N_samples)
                @test isapprox(val, 78.125, rtol=5e-3)
            end
        end
    end

    @testset "SortTransform" begin

        @testset "d = 1, constant integrand" begin
            let d = 1, c = contour, f = t -> 1.0, N = 2^17
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, -0.4, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1, c = contour, f = t -> t[1].val, N = 2^18
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, -0.28, rtol=1e-5)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, -0.905, rtol=1e-5)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, -8.405, rtol=1e-5)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, -12.78, rtol=1e-5)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, -12.5, rtol=1e-5)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2,
                c = contour,
                f = t -> (@assert kd.heaviside(c, t[1], t[2]); 1.0),
                N = 2^18
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, 0.08, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, -0.095+0.9im, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, -7.595+3.6im, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, -12.42+2.0im, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2,
                c = contour,
                f = t -> (@assert kd.heaviside(c, t[1], t[2]); t[1].val * t[2].val),
                N = 2^19
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, 0.0392, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, 0.409513, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, 35.322, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, 81.6642, rtol=1e-4)
                val = contour_integral(f, c, SortTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, 78.125, rtol=1e-4)
            end
        end
    end

    @testset "RootTransform" begin

        @testset "d = 1, constant integrand" begin
            let d = 1, c = contour, f = t -> 1.0, N = 2^17
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, -0.4, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1, c = contour, f = t -> t[1].val, N = 2^18
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, -0.28, rtol=1e-5)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, -0.905, rtol=1e-5)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, -8.405, rtol=1e-5)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, -12.78, rtol=1e-5)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, -12.5, rtol=1e-5)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2,
                c = contour,
                f = t -> (@assert kd.heaviside(c, t[1], t[2]); 1.0),
                N = 2^18
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, 0.08, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, -0.095+0.9im, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, -7.595+3.6im, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, -12.42+2.0im, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2,
                c = contour,
                f = t -> (@assert kd.heaviside(c, t[1], t[2]); t[1].val * t[2].val),
                N = 2^19
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(0.5)), N=N)
                @test isapprox(val, 0.0392, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(2.0)), N=N)
                @test isapprox(val, 0.409513, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(5.0)), N=N)
                @test isapprox(val, 35.322, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.5)), N=N)
                @test isapprox(val, 81.6642, rtol=1e-4)
                val = contour_integral(f, c, RootTransform(d, c, c(0.1), c(6.9)), N=N)
                @test isapprox(val, 78.125, rtol=1e-4)
            end
        end
    end

    @testset "DoubleSimplexRootTransform" begin

        @testset "d_before = 0, d_after = 3, constant integrand" begin
            let d_before = 0,
                d_after = 3,
                c = contour,
                t_i = c(1.1),
                t_w = c(5.0),
                t_f = c(5.5),
                N = 2^19
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t[3])
                    @assert kd.heaviside(c, t[3], t_w)
                    return 1.0
                end
                trans = DoubleSimplexRootTransform(d_before, d_after, c, t_i, t_w, t_f)
                val = contour_integral(f, c, trans, N=N)
                @test isapprox(val, 0.0208333im, rtol=1e-5)
            end
        end

        @testset "d_before = 0, d_after = 3, multilinear integrand" begin
            let d_before = 0,
                d_after = 3,
                c = contour,
                t_i = c(1.1),
                t_w = c(5.0),
                t_f = c(5.5),
                N = 2^19
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t[3])
                    @assert kd.heaviside(c, t[3], t_w)
                    return mapreduce(tau -> tau.val, *, t)
                end
                trans = DoubleSimplexRootTransform(d_before, d_after, c, t_i, t_w, t_f)
                val = contour_integral(f, c, trans, N=N)
                @test isapprox(val, -1.59928, rtol=1e-5)
            end
        end

        @testset "d_before = 3, d_after = 2, constant integrand" begin
            let d_before = 3,
                d_after = 2,
                c = contour,
                t_i = c(1.1),
                t_w = c(5.0),
                t_f = c(5.5),
                N = 2^20
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t_w)
                    @assert kd.heaviside(c, t_w, t[3])
                    @assert kd.heaviside(c, t[3], t[4])
                    @assert kd.heaviside(c, t[4], t[5])
                    @assert kd.heaviside(c, t[5], t_i)
                    return 1.0
                end
                trans = DoubleSimplexRootTransform(d_before, d_after, c, t_i, t_w, t_f)
                val = contour_integral(f, c, trans, N=N)
                @test isapprox(val, -1.23581im, rtol=1e-5)
            end
        end

        @testset "d_before = 3, d_after = 2, multilinear integrand" begin
            let d_before = 3,
                d_after = 2,
                c = contour,
                t_i = c(1.1),
                t_w = c(5.0),
                t_f = c(5.5),
                N = 2^20
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t_w)
                    @assert kd.heaviside(c, t_w, t[3])
                    @assert kd.heaviside(c, t[3], t[4])
                    @assert kd.heaviside(c, t[4], t[5])
                    @assert kd.heaviside(c, t[5], t_i)
                    return mapreduce(tau -> tau.val, *, t)
                end
                trans = DoubleSimplexRootTransform(d_before, d_after, c, t_i, t_w, t_f)
                val = contour_integral(f, c, trans, N=N)
                @test isapprox(val, -192.306, rtol=1e-5)
            end
        end
    end
end
