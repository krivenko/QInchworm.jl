using Test

import Keldysh; kd = Keldysh

import QInchworm.qmc_integrate: get_ref, qmc_time_ordered_integral

@testset "qmc_integrate" begin

    tmax = 1.
    β = 5.

    # -- Real-time Kadanoff-Baym contour
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));

    # ---------------
    # -- get_ref() --
    # ---------------

    for ref in [0.0, 0.5, 2.0, 5.0, 5.5, 6.5]
        @test get_ref(contour, contour(ref)) == ref
    end

    # ---------------------------------
    # -- qmc_time_ordered_integral() --
    # ---------------------------------

    τ = 5.0

    # d = 1, constant integrand
    let d = 1, f = t -> 1.0, c = contour, N = 200000
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(0.5), τ, d, N),
                       -0.4, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(2.0), τ, d, N),
                       -0.9-1.0im, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(5.0), τ, d, N),
                       -0.9-4.0im, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.5), τ, d, N),
                       -0.4-5.0im, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.9), τ, d, N),
                       -5.0im, rtol=1e-4)
    end

    # d = 1, linear in t integrand
    let d = 1, f = t -> t[1].val, c = contour, N = 200000
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(0.5), τ, d, N),
                       -0.28, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(2.0), τ, d, N),
                       -0.905, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(5.0), τ, d, N),
                       -8.405, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.5), τ, d, N),
                       -12.78, rtol=1e-4)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.9), τ, d, N),
                       -12.5, rtol=1e-4)
    end

    # d = 2, constant integrand
    let d = 2, f = t -> 1.0, c = contour, N = 500000
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(0.5), τ, d, N),
                       0.08, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(2.0), τ, d, N),
                       -0.095+0.9im, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(5.0), τ, d, N),
                       -7.595+3.6im, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.5), τ, d, N),
                       -12.42+2.0im, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.9), τ, d, N),
                       -12.5, rtol=5e-3)
    end

    # d = 2, bilinear in t1, t2 integrand
    let d = 2, f = t -> t[1].val * t[2].val, c = contour, N = 1000000
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(0.5), τ, d, N),
                       0.0392, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(2.0), τ, d, N),
                       0.409513, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(5.0), τ, d, N),
                       35.322, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.5), τ, d, N),
                       81.6642, rtol=5e-3)
        @test isapprox(qmc_time_ordered_integral(f, c, c(0.1), c(6.9), τ, d, N),
                       78.125, rtol=5e-3)
    end
end
