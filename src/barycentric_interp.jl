module barycentric_interp

using LinearAlgebra: ldiv!, mul!
using Keldysh: TimeGridPoint, BranchPoint, AbstractTimeGF, imaginary_branch
using QInchworm.ppgf: ImaginaryTimePPGFSector

export barycentric_interpolate!

function barycentric_interpolate!(x::Matrix{ComplexF64}, order::Int64, P::ImaginaryTimePPGFSector, t1::BranchPoint, t2::BranchPoint)
    @assert t1.domain == imaginary_branch
    @assert t2.domain == imaginary_branch
    
    Δt = t1.val - t2.val
    return barycentric_interpolate!(x, order, P, Δt)
end

barycentric_interpolate!(x::Matrix{ComplexF64}, order::Int64, P::ImaginaryTimePPGFSector, t1::TimeGridPoint, t2::TimeGridPoint) = barycentric_interpolate!(x, P, order, t1.bpoint, t2.bpoint)

function barycentric_interpolate!(x::Matrix{ComplexF64}, order::Int64, P::ImaginaryTimePPGFSector, t::ComplexF64)

    n = order # interpolation order

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

end # module barycentric_interp
