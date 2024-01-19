
using PyPlot; plt = PyPlot

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators
using QInchworm.ppgf: atomic_ppgf, partition_function

using QInchworm.barycentric_interp: barycentric_interpolate!

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
    
