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
# Authors: Hugo U. R. Strand, Igor Krivenko

using Test

using LinearAlgebra: I

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators
using QInchworm.ppgf: atomic_ppgf, partition_function

using QInchworm.barycentric_interp: barycentric_interpolate!

@testset "Barycentric interpolation" begin
    xi = [0., 1., 2., 3., 4.]
    fi = reshape((xi.-1).^2, (1, 1, length(xi)))
    
    f = Array{Float64, 2}(undef, 1, 1)
    for x in -1.0:0.1:5.0
        barycentric_interpolate!(f, x, xi, fi)
        @test f[1, 1] ≈ (x-1.)^2
    end
end

@testset "PPGF Barycentric interpolation" begin
    
    β = 5.0
    nτ = 100 + 1
    μ = 1.337
    
    H = μ * op.n(1)
    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H, soi)
    
    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)
    P = atomic_ppgf(grid, ed)    
    P_ana = atomic_ppgf(β, ed)
    
    τ_j = 0.:0.01:β
    p2_lin = Array{ComplexF64, 3}(undef, 1, 1, length(τ_j))
    p2_bary = Array{ComplexF64, 3}(undef, 1, 1, length(τ_j))
    p2_ana = Array{ComplexF64, 3}(undef, 1, 1, length(τ_j))
    mat = Matrix{ComplexF64}(undef, 1, 1)
    
    for (idx, τ) in enumerate(τ_j)
        t1 = BranchPoint(-im * τ, τ/β, kd.imaginary_branch)
        t2 = BranchPoint(0., 0., kd.imaginary_branch)   

        p2_lin[:, :, idx] = P[2](t1, t2)
        p2_ana[:, :, idx] = P_ana[2](t1, t2)

        order = 4
        barycentric_interpolate!(mat, order, P[2], t1, t2)
        p2_bary[:, :, idx] = mat
    end
    
    diff_lin = maximum(abs.((p2_lin - p2_ana) ./ p2_ana))
    @show diff_lin
    @test diff_lin < 1e-3

    diff_bary = maximum(abs.((p2_bary - p2_ana) ./ p2_ana))
    @show diff_bary
    @test diff_bary < 1e-6
    
    if false
        using PyPlot; plt = PyPlot
        plt.figure(figsize=(10, 10))
        subp = [3, 1, 1]
        
        plt.subplot(subp...); subp[end] += 1
        τ_i = [ -imag(p.bpoint.val) for p in grid.points ]
        plt.plot(τ_j, -imag(p2_lin[1, 1, :]), "-", label="linear")
        plt.plot(τ_j, -imag(p2_bary[1, 1, :]), "-", label="barycentric")
        plt.plot(τ_j, -imag(p2_ana[1, 1, :]), "--", label="analytic", zorder=10)
        plt.plot(τ_i, -imag(P[2].mat.data[1, 1, :]), "o", label="samples")
        plt.legend()
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$P(\tau)$")
        
        plt.subplot(subp...); subp[end] += 1
        plt.plot(τ_j, -imag(p2_lin - p2_ana)[1, 1, :], "-", label="linear")
        plt.plot(τ_j, -imag(p2_bary - p2_ana)[1, 1, :], "-", label="barycentric")
        plt.legend()
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$\Delta P(\tau)$")
        
        plt.subplot(subp...); subp[end] += 1
        plt.plot(τ_j, abs.((p2_lin - p2_ana) ./ p2_ana)[1, 1, :], "-", label="linear")
        plt.plot(τ_j, abs.((p2_bary - p2_ana) ./ p2_ana)[1, 1, :], "-", label="barycentric")
        plt.legend()
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$|\Delta P(\tau) / P(\tau)|$")
        plt.semilogy([], [])
        
        plt.tight_layout()
        plt.show()
    end

end
