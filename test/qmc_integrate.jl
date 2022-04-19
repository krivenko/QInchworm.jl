using Test

import Keldysh; kd = Keldysh

import QInchworm.qmc_integrate: get_ref,
                                qmc_time_ordered_integral,
                                qmc_time_ordered_integral_n_samples,
                                qmc_time_ordered_integral_sort

@testset verbose=true "qmc_integrate" begin

    tmax = 1.
    β = 5.

    # -- Real-time Kadanoff-Baym contour
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));

    @testset "get_ref()" begin
        for ref in [0.0, 0.5, 2.0, 5.0, 5.5, 6.5]
            @test get_ref(contour, contour(ref)) == ref
        end
    end

    @testset "qmc_time_ordered_integral()" begin

        τ = 5.0

        @testset "d = 1, constant integrand" begin
            let d = 1, f = t -> 1.0, c = contour, N = 200000
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(0.5); τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -0.4, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(2.0); τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(5.0); τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.5); τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.9); τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1, f = t -> t[1].val, c = contour, N = 200000
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(0.5), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -0.28, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(2.0), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -0.905, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(5.0), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -8.405, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.5), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -12.78, rtol=1e-4)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.9), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2, f = t -> 1.0, c = contour, N = 500000
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(0.5), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, 0.08, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(2.0), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -0.095+0.9im, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(5.0), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -7.595+3.6im, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.5), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -12.42+2.0im, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.9), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, -12.5, rtol=5e-3)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2, f = t -> t[1].val * t[2].val, c = contour, N = 1000000
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(0.5), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, 0.0392, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(2.0), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, 0.409513, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(5.0), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, 35.322, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.5), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, 81.6642, rtol=5e-3)
                val, N_samples = qmc_time_ordered_integral(f, d, c, c(0.1), c(6.9), τ=τ, N=N)
                @show N_samples / N
                @test isapprox(val, 78.125, rtol=5e-3)
            end
        end
    end

    @testset "qmc_time_ordered_integral_n_samples()" begin

        τ = 5.0

        @testset "d = 1, constant integrand" begin
            let d = 1, f = t -> 1.0, c = contour, N_samples = 50000
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(0.5); τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -0.4, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(2.0); τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(5.0); τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.5); τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.9); τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1, f = t -> t[1].val, c = contour, N_samples = 50000
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(0.5), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -0.28, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(2.0), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -0.905, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(5.0), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -8.405, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.5), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -12.78, rtol=1e-4)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.9), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2, f = t -> 1.0, c = contour, N_samples = 50000
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(0.5), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, 0.08, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(2.0), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -0.095+0.9im, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(5.0), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -7.595+3.6im, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.5), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -12.42+2.0im, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.9), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, -12.5, rtol=5e-3)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2, f = t -> t[1].val * t[2].val, c = contour, N_samples = 50000
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(0.5), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, 0.0392, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(2.0), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, 0.409513, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(5.0), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, 35.322, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.5), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, 81.6642, rtol=5e-3)
                val, N = qmc_time_ordered_integral_n_samples(f, d, c, c(0.1), c(6.9), τ=τ, N_samples=N_samples)
                @show N_samples / N
                @test isapprox(val, 78.125, rtol=5e-3)
            end
        end
    end

    @testset "qmc_time_ordered_integral_sort()" begin

        @testset "d = 1, constant integrand" begin
            let d = 1, f = t -> 1.0, c = contour, N = 200000
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(0.5); N=N)
                @test isapprox(val, -0.4, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(2.0); N=N)
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(5.0); N=N)
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.5); N=N)
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.9); N=N)
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1, f = t -> t[1].val, c = contour, N = 200000
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(0.5), N=N)
                @test isapprox(val, -0.28, rtol=1e-5)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(2.0), N=N)
                @test isapprox(val, -0.905, rtol=1e-5)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(5.0), N=N)
                @test isapprox(val, -8.405, rtol=1e-5)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.5), N=N)
                @test isapprox(val, -12.78, rtol=1e-5)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.9), N=N)
                @test isapprox(val, -12.5, rtol=1e-5)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2, f = t -> 1.0, c = contour, N = 500000
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(0.5), N=N)
                @test isapprox(val, 0.08, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(2.0), N=N)
                @test isapprox(val, -0.095+0.9im, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(5.0), N=N)
                @test isapprox(val, -7.595+3.6im, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.5), N=N)
                @test isapprox(val, -12.42+2.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.9), N=N)
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2, f = t -> t[1].val * t[2].val, c = contour, N = 1000000
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(0.5), N=N)
                @test isapprox(val, 0.0392, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(2.0), N=N)
                @test isapprox(val, 0.409513, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(5.0), N=N)
                @test isapprox(val, 35.322, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.5), N=N)
                @test isapprox(val, 81.6642, rtol=1e-4)
                val = qmc_time_ordered_integral_sort(f, d, c, c(0.1), c(6.9), N=N)
                @test isapprox(val, 78.125, rtol=1e-4)
            end
        end
    end
end
