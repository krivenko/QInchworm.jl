module analytic_ppgf

using LinearAlgebra: Diagonal, tr

using Keldysh: BranchPoint, AbstractTimeGF
using KeldyshED: EDCore, energies

import Keldysh: interpolate!
import KeldyshED: partition_function
import QInchworm.ppgf: partition_function, atomic_ppgf, density_matrix

export AtomicPPGF, partition_function, atomic_ppgf, density_matrix, interpolate!
    
struct AtomicPPGF <: AbstractTimeGF{ComplexF64, false}
    β::Float64
    E::Array{Float64, 1}
end

function (P::AtomicPPGF)(t::Number)
    return -im * Diagonal(exp.(-im * t * P.E))
end

function (P::AtomicPPGF)(t1::BranchPoint, t2::BranchPoint)
    Δt = t1.val - t2.val
    return P(Δt)
end

function interpolate!(x::Matrix{ComplexF64}, P::AtomicPPGF, t1::BranchPoint, t2::BranchPoint)
    Δt = t1.val - t2.val
    x[:] = P(Δt)
end

function atomic_ppgf(β::Float64, ed::EDCore)::Vector{AtomicPPGF}
    Z = partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[i P(β)] = Tr[ρ] = 1)
    P = [ AtomicPPGF(β, E .+ λ) for E in energies(ed) ]
    return P
end

function partition_function(P::Vector{AtomicPPGF})
    return sum(P, init=0im) do P_s
        im * tr(P_s(-im * P_s.β))
    end
end

function density_matrix(P::Vector{AtomicPPGF})::Vector{Matrix{ComplexF64}}
    @assert !isempty(P)
    return [1im * P_s(-im * P_s.β) for P_s in P]
end

# --

struct ScalarAnalyticGF{T, scalar} <: AbstractTimeGF{T, scalar}
    D::Float64
    E::Float64
end

function (P::ScalarAnalyticGF)(t::Number)
    return -im * exp.(-im * t * P.E) * P.D
end

function (P::ScalarAnalyticGF)(t1::BranchPoint, t2::BranchPoint)
    Δt = t1.val - t2.val
    return P(Δt)
end

function interpolate!(x::Matrix{ComplexF64}, P::ScalarAnalyticGF, t1::BranchPoint, t2::BranchPoint)
    Δt = t1.val - t2.val
    x[:] = P(Δt)
end

function analytic_gf(β::Float64, ϵ::Float64, C::Float64 = 1)::ScalarAnalyticGF{ComplexF64, true}
    D = C * 1/(1 + exp(-β*ϵ))
    return ScalarAnalyticGF{ComplexF64, true}(D, ϵ)
end


end # module analytic_ppgf
