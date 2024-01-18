

using LinearAlgebra: ldiv!, mul!

#using Keldysh: BranchPoint, AbstractTimeGF

#function barycentric_interpolate!(x::Matrix{ComplexF64}, P::PPGF, t1::BranchPoint, t2::BranchPoint)
#    Δt = t1.val - t2.val
#    return barycentric_interpolate!(x, P, Δt)
#end


#function barycentric_interpolate!(x::Matrix{ComplexF64}, P::PPGF, t::ComplexF64)
#    pass
#end


function equidistant_barycentric_weights(n::I)::Vector{I} where {I <: Integer}
    i = 0:n
    return (-1).^i .* binomial.(n, i)
end

function barycentric_interpolate!(f::Matrix{T}, x::S, xi::Vector{S}, fi::Array{T, 3}) where {T, S}
    n = length(xi)
    wi = equidistant_barycentric_weights(n - 1)
    barycentric_interpolate!(f, x, xi, fi, wi)
end

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

using PyPlot; plt = PyPlot

plt.plot(xi, fi[1, 1, :], "o")
plt.plot(xj, fj[1, 1, :], "-")
plt.show()
    
