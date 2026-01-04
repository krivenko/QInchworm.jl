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
# Author: Igor Krivenko

module DelaunaySimplex

export triangulate
export integrate_t3

using PyCall; py = PyCall

function nested_for_loop(f, d::Int, first::Int, last::Int, indices = Int[])
    if d == 0
        f(indices)
    else
        for i = first:last
            push!(indices, i)
            nested_for_loop(f, d - 1, i, last, indices)
            pop!(indices)
        end
    end
end

"""
    Delaunay triangulation of a d-dimensional regular simplex.

    `grid` Coordinates of points spanning an edge of the simplex.
"""
function triangulate(grid::AbstractArray{Float64, 1}, d::Int)
    points = Array{Float64, 2}(undef, d, 0)
    nested_for_loop(d, 1, length(grid)) do indices
        points = cat(points, grid[indices]; dims=2)
    end

    sp = py.pyimport("scipy.spatial")
    dl = sp.Delaunay(points', furthest_site=false)
    (points, py.py"$dl.simplices")
end

"""
    Use Delaunay triangulation and quadpy to integrate over a tetrahedron.
"""
function integrate_t3(integrand, points, simplices, order)
    qp = py.pyimport("quadpy")
    np = py.pyimport("numpy")

    calc = qp.t3.get_good_scheme(order)

    val = 0
    for i in axes(simplices, 1)
        vidx = simplices[i, :]
        tetra = Array{Float64}(Base.undef, 3, 0)
        for j = vidx
            tetra = cat(tetra, points[:, 1+j]; dims=2)
        end
        val += calc.integrate(integrand, np.array(tetra'))
    end
    val
end

end # module DelaunaySimplex
