
#using PyPlot; plt = PyPlot

using Test

using QInchworm.ppgf: atomic_ppgf, partition_function

using LinearAlgebra: Diagonal

using Keldysh; kd = Keldysh;
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.atomic_ppgf: analytic_atomic_ppgf, analytic_partition_function

@testset "atomic_ppgf" begin
    
    β = 5.0
    nτ = 11
    μ = 1.337
    
    H = μ * op.n(1)
    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H, soi)

    P = analytic_atomic_ppgf(ed, β)
    Z = analytic_partition_function(P, β)
    @test Z ≈ 1.0
    
    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)
    P0 = atomic_ppgf(grid, ed)
    Z0 = partition_function(P0)
    @test Z == Z0
    
    t0 = grid.points[1].bpoint
    
    for t1 in grid.points
        t1 = t1.bpoint
        for s in length(P)
            @test P[s](t1, t0) == P0[s](t1, t0)
        end
    end

    if false
        τ = [ -imag(t.bpoint.val) for t in grid.points ]

        p1 = Array{ComplexF64, 3}(undef, 1, 1, nτ)
        p2 = Array{ComplexF64, 3}(undef, 1, 1, nτ)
        
        t0 = grid.points[1].bpoint
        for (idx, t1) in enumerate(grid.points)
            t1 = t1.bpoint
            p1[:, :, idx] = P[1](t1, t0)
            p2[:, :, idx] = P[2](t1, t0)
        end

        plt.plot(τ, -imag(P0[1].mat.data[1, 1, :]), "x-")
        plt.plot(τ, -imag(P0[2].mat.data[1, 1, :]), "x-")
        
        plt.plot(τ, -imag(p1[1, 1, :]), "+-")
        plt.plot(τ, -imag(p2[1, 1, :]), "+-")
        
        plt.ylim([-0.1, 1.1])
        plt.show()
    end

end # testset
