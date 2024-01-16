using Test

using QInchworm.keldysh_dlr: DLRImaginaryTimeGrid, DLRImaginaryTimeGF

import Lehmann; le = Lehmann
using Keldysh; kd = Keldysh

using QuadGK: quadgk

using LinearAlgebra: diag, tr, I


function semi_circular_g_tau(times, t, h, β)

    g_out = zero(times)

    function kernel(t, w)
        if w > 0
            return exp(-t * w) / (1 + exp(-w))
        else
            return exp((1 - t)*w) / (1 + exp(w))
        end
    end

    for (i, τ) in enumerate(times)
        I = x -> -2 / pi / t^2 * kernel(τ/β, β*x) * sqrt(x + t - h) * sqrt(t + h - x)
        g, err = quadgk(I, -t+h, t+h; rtol=1e-12)
        g_out[i] = g
    end

    return g_out
end


@testset "Keldysh DLR gf" begin
    
    β = 10.0
    dlr = le.DLRGrid(Euv=1., β=β, isFermi=true, rtol=1e-10, rebuild=true, verbose=false)
    @show length(dlr.τ)
    
    contour = kd.ImaginaryContour(β=β)
    grid = DLRImaginaryTimeGrid(contour, dlr)

    dos = kd.bethe_dos(t=0.5, ϵ=0.0)
    g = DLRImaginaryTimeGF(dos, grid)
    
    # -- Interpolate!
    
    using Random
    rng = MersenneTwister(1234)
    t_rand = zeros(10)
    rand!(rng, t_rand)
    t_rand .*= dlr.β;
    
    G_t_rand = vec(le.dlr2tau(dlr, -im * g.mat.data, t_rand, axis=3))
    G_t_rand_ref = semi_circular_g_tau(t_rand, 1.0, 0.0, dlr.β)
    
    diff = maximum(abs.(G_t_rand - G_t_rand_ref))
    @show diff
    @test diff < 1e-10
    
    # -- Interpolate API of Keldysh.jl
    
    τ = 0.1 * β
    bp = BranchPoint(im * τ, τ/β, kd.imaginary_branch)
    bp0 = BranchPoint(0., 0., kd.imaginary_branch)
    
    res = g(bp, bp0)
    #@show res
    
    ref = im * semi_circular_g_tau([τ], 1.0, 0.0, dlr.β)[1]
    #@show ref
    
    @test res ≈ ref

end # testset
