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

# Test exact atomic pseudo particle Green's function (exact_atomic_ppgf) module
# by comparing with the standard equidistant constructor in the ppgf module.

using Test

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators

using QInchworm.ppgf: atomic_ppgf, partition_function
using QInchworm.exact_atomic_ppgf: exact_atomic_ppgf, partition_function

@testset "exact_atomic_ppgf" begin

    β = 5.0
    nτ = 11
    μ = 1.337

    H = μ * op.n(1)
    soi = ked.Hilbert.SetOfIndices([[1]])
    ed = ked.EDCore(H, soi)

    P = atomic_ppgf(β, ed)
    Z = partition_function(P)
    @test Z ≈ 1.0

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)
    P0 = atomic_ppgf(grid, ed)
    Z0 = partition_function(P0)
    @test Z == Z0

    t0 = grid.points[1].bpoint

    for t1 in grid.points
        t1 = t1.bpoint
        for (p, p0) in zip(P, P0)
            @test p(t1, t0) == p0(t1, t0)
        end
    end

end
