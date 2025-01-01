# QInchworm.jl
#
# Copyright (C) 2021-2025 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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

"""
Extension of Keldysh.jl defining imaginary time Green's functions
represented using the Discrete Lehmann Representation as implemented
in Lehmann.jl.

# Exports
$(EXPORTS)
"""
module keldysh_dlr

using DocStringExtensions

using Lehmann; le = Lehmann

using Keldysh; kd = Keldysh
using Keldysh: AbstractTimeGrid
using Keldysh: ImaginaryContour, TimeGridPoint, PeriodicStorage, GFSignEnum, BranchPoint

import QInchworm.utility: ph_conj

export DLRImaginaryTimeGrid, DLRImaginaryTimeGF
export ph_conj

#
# DLRImaginaryTimeGrid
#

"""
    $(TYPEDEF)

Wrapper around Lehmann.jl describing a Discrete Lehmann Representation imaginary time grid
conforming to the interface of Keldysh.jl TimeGrids.

# Fields
$(TYPEDFIELDS)
"""
struct DLRImaginaryTimeGrid <: AbstractTimeGrid
    contour::ImaginaryContour
    points::Vector{TimeGridPoint}
    branch_bounds::NTuple{1, Pair{TimeGridPoint, TimeGridPoint}}
    ntau::Int
    dlr::le.DLRGrid

    function DLRImaginaryTimeGrid(c::ImaginaryContour, dlr::le.DLRGrid)
        points::Vector{TimeGridPoint} = []
        τ_branch = c.branches[1]
        @assert τ_branch.domain == kd.imaginary_branch
        for (idx, τ) in enumerate(dlr.τ)
            point = TimeGridPoint(idx, idx, τ_branch(τ/dlr.β))
            push!(points, point)
        end
        τ_0 = TimeGridPoint(-1, -1, τ_branch(0.))
        τ_β = TimeGridPoint(length(points) + 1, length(points) + 1, τ_branch(1.))

        branch_bounds = ( Pair(τ_0, τ_β), )
        ntau = length(dlr.τ)
        return new(c, points, branch_bounds, ntau, dlr)
    end
end

Base.:isequal(A::T, B::T) where T <: DLRImaginaryTimeGrid = all(A.points .== B.points)

"""
    $(TYPEDEF)

Wrapper around Lehmann.jl describing a Discrete Lehmann Representation imaginary time Green's
function conforming to the interface of Keldysh.jl AbstractTimeGF.

# Fields
$(TYPEDFIELDS)
"""
struct DLRImaginaryTimeGF{T, scalar} <: AbstractTimeGF{T, scalar}
    grid::DLRImaginaryTimeGrid
    mat::PeriodicStorage{T,scalar}
    ξ::GFSignEnum
end

"""
    $(TYPEDSIGNATURES)

Make a [`DLRImaginaryTimeGF`](@ref) from a [`DLRImaginaryTimeGrid`](@ref)
following the API of Keldysh.ImaginarTimeGF.

"""
function DLRImaginaryTimeGF(::Type{T},
                            grid::DLRImaginaryTimeGrid,
                            norb=1,
                            ξ::GFSignEnum=fermionic,
                            scalar=false) where T <: Number
    ntau = grid.ntau
    mat = PeriodicStorage(T, ntau, norb, scalar)
    return DLRImaginaryTimeGF(grid, mat, ξ)
end

DLRImaginaryTimeGF(grid::DLRImaginaryTimeGrid,
                   norb=1,
                   ξ::GFSignEnum=fermionic,
                   scalar=false) = DLRImaginaryTimeGF(ComplexF64, grid, norb, ξ, scalar)

norbitals(G::DLRImaginaryTimeGF) = G.mat.norb

function _compatible(A::T, B::T) where T <: DLRImaginaryTimeGF
    A.grid == B.grid || throw(DimensionMismatch("The DLR grids of the two Green's functions differ."))
    A.ξ == B.ξ || throw(DomainError("The statistics of the two Green's functions differ."))
    return true
end

#
# Arithmetic operations
#

function Base.:+(A::T, B::T) where T <: DLRImaginaryTimeGF
    @assert _compatible(A, B)
    return T(A.grid, A.mat + B.mat, A.ξ)
end

function Base.:-(A::T, B::T) where T <: DLRImaginaryTimeGF
    @assert _compatible(A, B)
    return T(A.grid, A.mat - B.mat, A.ξ)
end

Base.:-(A::T) where T <: DLRImaginaryTimeGF = T(A.grid, -A.mat, A.ξ)

Base.:*(G::T, α::Number) where T <: DLRImaginaryTimeGF = T(G.grid, α * G.mat, G.ξ)
Base.:*(α::Number, G::T) where T <: DLRImaginaryTimeGF = G * α

function Base.:isapprox(A::T, B::T) where T <: DLRImaginaryTimeGF
    @assert _compatible(A, B)
    return A.mat.data ≈ B.mat.data
end

#
# Matrix valued Gf interpolator interface
#

function (G::DLRImaginaryTimeGF{T, false})(z1::BranchPoint, z2::BranchPoint) where T
  norb = norbitals(G)
  x = zeros(T, norb, norb)
  return interpolate!(x, G, z1, z2)
end

function interpolate!(x, G::DLRImaginaryTimeGF{T, false},
                      z1::BranchPoint, z2::BranchPoint) where T
    Δτ, sign = _Δτ_and_sign(G, z1, z2)
    x[:] = le.dlr2tau(G.grid.dlr, sign * g.mat.data, [Δτ], axis=3)
    return x
end

#
# Scalar valued Gf interpolator interface
#

function (G::DLRImaginaryTimeGF{T, true})(z1::BranchPoint, z2::BranchPoint) where T
    return interpolate(G, z1, z2)
end

function interpolate(G::DLRImaginaryTimeGF{T, true},
                     z1::BranchPoint, z2::BranchPoint)::T where T
    Δτ, sign = _Δτ_and_sign(G, z1, z2)
    return le.dlr2tau(G.grid.dlr, sign * G.mat.data, [Δτ], axis=3)[1, 1, 1]
end

# Interpolation helper function

function _Δτ_and_sign(G::DLRImaginaryTimeGF{T, true},
                     z1::BranchPoint, z2::BranchPoint) where T

    @assert z1.domain == kd.imaginary_branch
    @assert z2.domain == kd.imaginary_branch

    sign = +1.0
    Δτ = -imag(z1.val - z2.val) # z = -iτ

    if !kd.heaviside(z1, z2) # !(z1 >= z2)
        Δτ = G.grid.dlr.β + Δτ
        sign = sign * Int(G.ξ)
    end

    @assert 0. <= Δτ <= G.grid.dlr.β

    return Δτ, sign
end

#
# Green's function creation from function
#

"""
    $(TYPEDSIGNATURES)

Make a [`DLRImaginaryTimeGF`](@ref) from a function

"""
DLRImaginaryTimeGF(f::Function,
                   grid::DLRImaginaryTimeGrid,
                   norb=1,
                   ξ::GFSignEnum=fermionic,
                   scalar=false) = DLRImaginaryTimeGF(f, ComplexF64, grid, norb, ξ, scalar)

function DLRImaginaryTimeGF(f::Function,
                            ::Type{T},
                            grid::DLRImaginaryTimeGrid,
                            norb=1,
                            ξ::GFSignEnum=fermionic,
                            scalar=false) where T <: Number
    G = DLRImaginaryTimeGF(T, grid, norb, ξ, scalar)

    t0 = TimeGridPoint(1, -1, BranchPoint(im * 0., 0., kd.imaginary_branch))

    g_τ = Array{ComplexF64, 3}(undef, norb, norb, length(grid.dlr.τ))
    for (idx, t) in enumerate(G.grid.points)
        if scalar
            g_τ[1, 1, idx] = f(t, t0)
        else
            g_τ[:, :, idx] = f(t, t0)
        end
    end

    G.mat.data[:] = le.tau2dlr(G.grid.dlr, g_τ, axis=3)

    return G
end

"""
    $(TYPEDSIGNATURES)

Make a [`DLRImaginaryTimeGF`](@ref) from a Keldysh.jl AbstractDOS object.

"""
function DLRImaginaryTimeGF(dos::AbstractDOS, grid::DLRImaginaryTimeGrid)
    β = grid.contour.β
    DLRImaginaryTimeGF(grid, 1, fermionic, true) do z1, z2
        kd.dos2gf(dos, β, z1.bpoint, z2.bpoint)
    end
end

"""
    $(TYPEDSIGNATURES)

Given a scalar-valued imaginary time Green's function ``g(\\tau)``, return its particle-hole
conjugate ``g(\\beta-\\tau)``.
"""
function ph_conj(g::DLRImaginaryTimeGF{T, true})::DLRImaginaryTimeGF{T, true} where {T}
    g_rev = deepcopy(g)

    dlr = g.grid.dlr
    g_τ = le.dlr2tau(dlr, g.mat.data, dlr.β .- dlr.τ, axis=3)
    g_rev.mat.data[:] = le.tau2dlr(dlr, g_τ, axis=3)

    return g_rev
end

end # module keldysh_dlr
