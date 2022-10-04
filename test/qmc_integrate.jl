using Test

import Keldysh; kd = Keldysh

import QInchworm.qmc_integrate: qmc_time_ordered_integral,
                                qmc_time_ordered_integral_n_samples,
                                qmc_time_ordered_integral_sort,
                                qmc_time_ordered_integral_root,
                                qmc_inchworm_integral_root

@testset verbose=true "qmc_integrate" begin

    tmax = 1.
    β = 5.

    # -- Real-time Kadanoff-Baym contour
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));

    @testset "qmc_time_ordered_integral()" begin

        τ = 5.0

        @testset "d = 1, constant integrand" begin
            let d = 1, c = contour, f = t -> 1.0, N = 200000
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
            let d = 1, c = contour, f = t -> t[1].val, N = 200000
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
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); 1.0), N = 500000
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
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); t[1].val * t[2].val), N = 1000000
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
            let d = 1, c = contour, f = t -> 1.0, N_samples = 50000
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
            let d = 1, c = contour, f = t -> t[1].val, N_samples = 50000
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
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); 1.0), N_samples = 50000
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
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); t[1].val * t[2].val), N_samples = 50000
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
            let d = 1, c = contour, f = t -> 1.0, N = 200000
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
            let d = 1, c = contour, f = t -> t[1].val, N = 200000
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
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); 1.0), N = 500000
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
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); t[1].val * t[2].val), N = 1000000
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

    @testset "qmc_time_ordered_integral_root()" begin

        @testset "d = 1, constant integrand" begin
            let d = 1, c = contour, f = t -> 1.0, N = 200000
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(0.5); N=N)
                @test isapprox(val, -0.4, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(2.0); N=N)
                @test isapprox(val, -0.9-1.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(5.0); N=N)
                @test isapprox(val, -0.9-4.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.5); N=N)
                @test isapprox(val, -0.4-5.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.9); N=N)
                @test isapprox(val, -5.0im, rtol=1e-4)
            end
        end

        @testset "d = 1, linear in t integrand" begin
            let d = 1, c = contour, f = t -> t[1].val, N = 200000
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(0.5), N=N)
                @test isapprox(val, -0.28, rtol=1e-5)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(2.0), N=N)
                @test isapprox(val, -0.905, rtol=1e-5)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(5.0), N=N)
                @test isapprox(val, -8.405, rtol=1e-5)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.5), N=N)
                @test isapprox(val, -12.78, rtol=1e-5)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.9), N=N)
                @test isapprox(val, -12.5, rtol=1e-5)
            end
        end

        @testset "d = 2, constant integrand" begin
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); 1.0), N = 500000
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(0.5), N=N)
                @test isapprox(val, 0.08, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(2.0), N=N)
                @test isapprox(val, -0.095+0.9im, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(5.0), N=N)
                @test isapprox(val, -7.595+3.6im, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.5), N=N)
                @test isapprox(val, -12.42+2.0im, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.9), N=N)
                @test isapprox(val, -12.5, rtol=1e-4)
            end
        end

        @testset "d = 2, bilinear in t1, t2 integrand" begin
            let d = 2, c = contour, f = t -> (@assert kd.heaviside(c, t[1], t[2]); t[1].val * t[2].val), N = 1000000
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(0.5), N=N)
                @test isapprox(val, 0.0392, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(2.0), N=N)
                @test isapprox(val, 0.409513, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(5.0), N=N)
                @test isapprox(val, 35.322, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.5), N=N)
                @test isapprox(val, 81.6642, rtol=1e-4)
                val = qmc_time_ordered_integral_root(f, d, c, c(0.1), c(6.9), N=N)
                @test isapprox(val, 78.125, rtol=1e-4)
            end
        end
    end

    @testset "qmc_inchworm_integral_root()" begin

        @testset "d_bold = 0, d_bare = 3, constant integrand" begin
            let d_bold = 0, d_bare = 3, c = contour, t_i = c(1.1), t_w = c(5.0), t_f = c(5.5), N = 1000000
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t[3])
                    @assert kd.heaviside(c, t[3], t_w)
                    1.0
                end
                val = qmc_inchworm_integral_root(f, d_bold, d_bare, c, t_i, t_w, t_f; N=N)
                @test isapprox(val, 0.0208333im, rtol=1e-5)
            end
        end

        @testset "d_bold = 0, d_bare = 3, multilinear integrand" begin
            let d_bold = 0, d_bare = 3, c = contour, t_i = c(1.1), t_w = c(5.0), t_f = c(5.5), N = 1000000
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t[3])
                    @assert kd.heaviside(c, t[3], t_w)
                    mapreduce(tau -> tau.val, *, t)
                end
                val = qmc_inchworm_integral_root(f, d_bold, d_bare, c, t_i, t_w, t_f; N=N)
                @test isapprox(val, -1.59928, rtol=1e-5)
            end
        end

        @testset "d_bold = 3, d_bare = 2, constant integrand" begin
            let d_bold = 3, d_bare = 2, c = contour, t_i = c(1.1), t_w = c(5.0), t_f = c(5.5), N = 2000000
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t_w)
                    @assert kd.heaviside(c, t_w, t[3])
                    @assert kd.heaviside(c, t[3], t[4])
                    @assert kd.heaviside(c, t[4], t[5])
                    @assert kd.heaviside(c, t[5], t_i)
                    1.0
                end
                val = qmc_inchworm_integral_root(f, d_bold, d_bare, c, t_i, t_w, t_f; N=N)
                @test isapprox(val, -1.23581im, rtol=1e-5)
            end
        end

        @testset "d_bold = 3, d_bare = 2, multilinear integrand" begin
            let d_bold = 3, d_bare = 2, c = contour, t_i = c(1.1), t_w = c(5.0), t_f = c(5.5), N = 2000000
                function f(t)
                    @assert kd.heaviside(c, t_f, t[1])
                    @assert kd.heaviside(c, t[1], t[2])
                    @assert kd.heaviside(c, t[2], t_w)
                    @assert kd.heaviside(c, t_w, t[3])
                    @assert kd.heaviside(c, t[3], t[4])
                    @assert kd.heaviside(c, t[4], t[5])
                    @assert kd.heaviside(c, t[5], t_i)
                    mapreduce(tau -> tau.val, *, t)
                end
                val = qmc_inchworm_integral_root(f, d_bold, d_bare, c, t_i, t_w, t_f; N=N)
                @test isapprox(val, -192.306, rtol=1e-5)
            end
        end
    end
end
