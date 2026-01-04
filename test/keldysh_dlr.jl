# QInchworm.jl
#
# Copyright (C) 2021-2026 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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

using Random

using QInchworm.keldysh_dlr: DLRImaginaryTimeGrid, DLRImaginaryTimeGF

using Lehmann; le = Lehmann
using Keldysh; kd = Keldysh

@testset "Keldysh DLR gf" begin

    t = 1.0
    ϵ = 0.1337
    β = 10.0
    dlr = le.DLRGrid(Euv=1., β=β, isFermi=true, rtol=1e-10, rebuild=true, verbose=false)

    contour = kd.ImaginaryContour(β=β)
    grid = DLRImaginaryTimeGrid(contour, dlr)

    dos = kd.bethe_dos(t=0.5*t, ϵ=ϵ)
    g = DLRImaginaryTimeGF(dos, grid)

    # Test existence of arithmetic ops
    G = g * 1
    G = 1 * G
    G = G + G
    G = G - 0.5 * G
    G = -G
    G = -1 * G

    @test g ≈ G

    # -- Interpolate!

    τ_branch = contour.branches[1]
    bp0 = τ_branch(0)

    rng = MersenneTwister(1234)
    t_rand = rand(rng, Float64, 10)

    G_t_rand = vec(le.dlr2tau(dlr, -im * g.mat.data, t_rand * β, axis=3))
    G_t_rand_ref = -im * [kd.dos2gf(dos, β, τ_branch(t), bp0) for t in t_rand]

    diff = maximum(abs.(G_t_rand - G_t_rand_ref))
    @test diff < 1e-10

    # -- Interpolate API of Keldysh.jl

    bp1 = τ_branch(0.3)
    bp2 = τ_branch(0.1)

    # test eval for positive time differences
    ref = kd.dos2gf(dos, β, bp1, bp2)
    res = g(bp1, bp2)
    @test res ≈ ref

    # test eval for negative time differences
    ref = kd.dos2gf(dos, β, bp2, bp1)
    res = g(bp2, bp1)
    @test res ≈ ref
    
end
