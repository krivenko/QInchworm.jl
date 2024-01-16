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

function (G::AbstractTimeGF{T, true})(t1::BranchPoint, t2::BranchPoint) where T
  return interpolate(G, t1, t2)
end

function interpolate(G::DLRImaginaryTimeGF{T, true}, t1::BranchPoint, t2::BranchPoint) where T
    dlr = G.grid.dlr
    τ = dlr.β*(t1.ref - t2.ref)
    return le.dlr2tau(dlr, g.mat.data, [τ], axis=3)
end

# ====================================================================

using PyCall

function semi_circular_g_tau(times, t, h, β)

    #np = PyCall.pyimport("numpy")
    kernel = PyCall.pyimport("pydlr").kernel
    quad = PyCall.pyimport("scipy.integrate").quad

    #def eval_semi_circ_tau(tau, beta, h, t):
    #    I = lambda x : -2 / np.pi / t**2 * kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
    #    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    #    return g

    g_out = zero(times)

    for (i, tau) in enumerate(times)
        I = x -> -2 / π / t^2 * kernel([tau/β], [β*x])[1, 1]
        g, res = quad(I, -t+h, t+h, weight="alg", wvar=(0.5, 0.5))
        g_out[i] = g
    end

    return g_out
end

# ====================================================================

β = 10.0
dlr = le.DLRGrid(Euv=8., β=β, isFermi=true, rtol=1e-12, rebuild=true, verbose=false)
@show size(dlr.τ)

contour = kd.ImaginaryContour(β=β)
grid = DLRImaginaryTimeGrid(contour, dlr)
g = DLRImaginaryTimeGF(grid, 1, kd.fermionic, true)
#@show g
@show size(g.mat.data)

g_τ = im * -2/π * le.Sample.SemiCircle(1.0, dlr.β, true, dlr.τ, :τ; degree=128)
g_c = le.tau2dlr(dlr, g_τ)

#g.mat.data[:] = im * -2/π * le.Sample.SemiCircle(1.0, dlr.β, true, dlr.τ, :τ; degree=128)
g.mat.data[:] = g_c

#@show size(g.mat.data)
#@show g

# Interpolate!

using Random
rng = MersenneTwister(1234)
t_rand = zeros(100)
rand!(rng, t_rand)
t_rand .*= dlr.β;

#G_t_rand = vec(le.tau2tau(dlr, -im * g.mat.data, t_rand, axis=3))
G_t_rand = vec(le.dlr2tau(dlr, -im * g.mat.data, t_rand, axis=3))
G_t_rand_ref = semi_circular_g_tau(t_rand, 1.0, 0.0, dlr.β)

diff = maximum(abs.(G_t_rand - G_t_rand_ref))
@show diff
@assert diff < 1e-10

# Interpolate API of Keldysh.jl

τ = 0.1 * β
bp = BranchPoint(im * τ, τ/β, kd.imaginary_branch)
bp0 = BranchPoint(0., 0., kd.imaginary_branch)

res = g(bp, bp0)
@show res

ref = im * semi_circular_g_tau([τ], 1.0, 0.0, dlr.β)
@show ref

@assert res ≈ ref
