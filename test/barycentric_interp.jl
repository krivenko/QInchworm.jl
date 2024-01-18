using PyPlot; plt = PyPlot

using LinearAlgebra: ldiv!, mul!

using Keldysh: BranchPoint, AbstractTimeGF, imaginary_branch
using QInchworm.expansion: AllPPGFSectorTypes

using QInchworm.ppgf: ImaginaryTimePPGFSector

# --

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators
using QInchworm.ppgf: atomic_ppgf, partition_function
    
# ----------------------------------------------------------------------------------

function barycentric_interpolate!(x::Matrix{ComplexF64}, order::Int64, P::ImaginaryTimePPGFSector, t1::BranchPoint, t2::BranchPoint)
    @assert t1.domain == imaginary_branch
    @assert t2.domain == imaginary_branch
    
    Δt = t1.val - t2.val
    return barycentric_interpolate!(x, order, P, Δt)
end

barycentric_interpolate!(x::Matrix{ComplexF64}, order::Int64, P::ImaginaryTimePPGFSector, t1::TimeGridPoint, t2::TimeGridPoint) = barycentric_interpolate!(x, P, order, t1.bpoint, t2.bpoint)

function barycentric_interpolate!(x::Matrix{ComplexF64}, order::Int64, P::ImaginaryTimePPGFSector, t::ComplexF64)

    #n = 4 # Order of interpolation
    n = order

    β = P.grid.contour.β
    nτ = P.grid.ntau
    τ = -imag(t)

    idx = 1 + (nτ - 1) * τ / β
    idx_l, idx_h = floor(Int64, idx), ceil(Int64, idx)

    if idx_l != idx_h
        i = idx >= n ? (idx_h-n+1:idx_h) : (1:n)
        @assert length(i) == n
        #i = idx >= n ? (1:idx_h) : (1:n)
        
        #@show length(i), i
        τ_i = [-imag(p.bpoint.val) for p in P.grid.points[i]]
        #@show τ_i
        barycentric_interpolate!(x, τ, τ_i, P.mat.data[:, :, i])
    else
        x[:] .= P.mat.data[:, :, idx_l]
    end
end

# ----------------------------------------------------------------------------------

"""
Barycentric interpolation of f_i = f(x_i) on equidistant nodes x_i.

- Assuming x_i is equidistant and sorted.

Note: Numerically unstable for large numer of nodes.

Formulas from:
Barycentric Lagrange Interpolation
Jean-Paul Berrut and Lloyd N. Trefethen, SIAM Review, v46, 3 (2004)
https://doi.org/10.1137/S0036144502417715
"""
function barycentric_interpolate!(f::Matrix{T}, x::S, xi::Vector{S}, fi::Array{T, 3}, wi::Vector{I}) where {T, S, I}

    @assert length(xi) == size(fi)[end]
    @assert size(f) == size(fi)[1:2]
    
    a, b, n = size(fi)
    idx = searchsortedfirst(xi, x)

    if idx <= n && x == xi[idx]
        f[:] = fi[:, :, idx]
        return
    end

    f_vec = reshape(f, (a*b))
    fi_mat = reshape(fi, (a*b, n))

    ri = wi ./ ( x .- xi )
    
    mul!(f_vec, fi_mat, ri, 1.0, 0.0)
    ldiv!(sum(ri), f)
end

function barycentric_interpolate!(f::Matrix{T}, x::S, xi::Vector{S}, fi::Array{T, 3}) where {T, S}
    n = length(xi)
    wi = equidistant_barycentric_weights(n - 1) # TODO: Store precomputed weights!
    barycentric_interpolate!(f, x, xi, fi, wi)
end

function equidistant_barycentric_weights(n::I)::Vector{I} where {I <: Integer}
    i = 0:n
    return (-1).^i .* binomial.(n, i)
end

# ----------------------------------------------------------------------------------


xi = [0., 1., 2., 3., 4.]
fi = reshape((xi.-1).^2, (1, 1, length(xi)))

@show xi
@show fi

#x = 1.5
#f = zeros((1, 1))
#barycentric_interpolate!(f, x, xi, fi)
#@show f

xj = -1.0:0.1:5.0
fj = zeros((1, 1, length(xj)))
for (j, x) in enumerate(xj)
    f = zeros((1, 1))
    barycentric_interpolate!(f, x, xi, fi)
    fj[:, :, j] = f
end

#@show fj

# --

β = 5.0
nτ = 10 + 1
μ = 1.337

H = μ * op.n(1)
soi = ked.Hilbert.SetOfIndices([[1]])
ed = ked.EDCore(H, soi)

contour = kd.ImaginaryContour(β=β)
grid = kd.ImaginaryTimeGrid(contour, nτ)
P = atomic_ppgf(grid, ed)
#@show P[2].mat.data[1, 1, :]

P_ana = atomic_ppgf(β, ed)

order = 4
τ_j = 0.:0.01:β
p2_lin = Array{ComplexF64, 3}(undef, 1, 1, length(τ_j))
p2_bary = Array{ComplexF64, 3}(undef, 1, 1, length(τ_j))
p2_ana = Array{ComplexF64, 3}(undef, 1, 1, length(τ_j))
mat = Matrix{ComplexF64}(undef, 1, 1)

for (idx, τ) in enumerate(τ_j)
    t1 = BranchPoint(-im * τ, τ/β, kd.imaginary_branch)
    t2 = BranchPoint(0., 0., kd.imaginary_branch)   
    barycentric_interpolate!(mat, order, P[2], t1, t2)
    #@show mat
    p2_lin[:, :, idx] = P[2](t1, t2)
    p2_bary[:, :, idx] = mat
    p2_ana[:, :, idx] = P_ana[2](t1, t2)
end

@show size(p2_lin)

# --
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

#plt.plot(xi, fi[1, 1, :], "o")
#plt.plot(xj, fj[1, 1, :], "-")

plt.tight_layout()
plt.show()
    
