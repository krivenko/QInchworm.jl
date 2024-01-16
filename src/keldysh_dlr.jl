module keldysh_dlr

import Lehmann; le = Lehmann;

using Keldysh; kd = Keldysh

import Keldysh: AbstractTimeGrid, ImaginaryContour, TimeGridPoint, PeriodicStorage, GFSignEnum, BranchPoint

struct DLRImaginaryTimeGrid <: AbstractTimeGrid
  contour::ImaginaryContour
  points::Vector{TimeGridPoint}
  branch_bounds::NTuple{1, Pair{TimeGridPoint, TimeGridPoint}}
  ntau::Int
  dlr::le.DLRGrid

  function DLRImaginaryTimeGrid(c::ImaginaryContour, dlr::le.DLRGrid)
    points::Vector{TimeGridPoint} = []
    for (idx, τ) in enumerate(dlr.τ)
        bpoint = BranchPoint(im * τ, τ/dlr.β, kd.imaginary_branch)
        push!(points, TimeGridPoint(1, idx, bpoint))
    end
    τ_0 = TimeGridPoint(1, -1, BranchPoint(0., 0., kd.imaginary_branch))
    τ_β = TimeGridPoint(1, -1, BranchPoint(im * dlr.β, 1., kd.imaginary_branch))
      
    branch_bounds = ( Pair(τ_0, τ_β), )
    ntau = length(dlr.τ)
    return new(c, points, branch_bounds, ntau, dlr)
  end
end

struct DLRImaginaryTimeGF{T, scalar} <: AbstractTimeGF{T, scalar}
  grid::DLRImaginaryTimeGrid
  mat::PeriodicStorage{T,scalar}
  ξ::GFSignEnum
end

function DLRImaginaryTimeGF(::Type{T}, grid::DLRImaginaryTimeGrid, norb=1, ξ::GFSignEnum=fermionic, scalar=false) where T <: Number
  ntau = grid.ntau
  mat = PeriodicStorage(T, ntau, norb, scalar)
  DLRImaginaryTimeGF(grid, mat, ξ)
end
DLRImaginaryTimeGF(grid::DLRImaginaryTimeGrid, norb=1, ξ::GFSignEnum=fermionic, scalar=false) = DLRImaginaryTimeGF(ComplexF64, grid, norb, ξ, scalar)

norbitals(G::DLRImaginaryTimeGF) = G.mat.norb

# Matrix valued Gf interpolator interface

function (G::DLRImaginaryTimeGF{T, false})(t1::BranchPoint, t2::BranchPoint) where T
  norb = norbitals(G)
  x = zeros(T, norb, norb)
  return interpolate!(x, G, t1, t2)
end

function interpolate!(x, G::DLRImaginaryTimeGF{T, false}, t1::BranchPoint, t2::BranchPoint) where T
    dlr = G.grid.dlr
    τ = dlr.β*(t1.ref - t2.ref)
    #x[:] = le.tau2tau(dlr, g.mat.data, [τ], axis=3)
    x[:] = le.dlr2tau(dlr, g.mat.data, [τ], axis=3)
    return x
end

# Scalar valued Gf interpolator interface

function (G::DLRImaginaryTimeGF{T, true})(t1::BranchPoint, t2::BranchPoint) where T
  return interpolate(G, t1, t2)
end

function interpolate(G::DLRImaginaryTimeGF{T, true}, t1::BranchPoint, t2::BranchPoint) where T
    dlr = G.grid.dlr
    τ = dlr.β*(t1.ref - t2.ref)
    return le.dlr2tau(dlr, G.mat.data, [τ], axis=3)
end

end # module keldysh_dlr
