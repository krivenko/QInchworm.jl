module atomic_ppgf

using LinearAlgebra: Diagonal, tr

using Keldysh: BranchPoint, AbstractTimeGF
using KeldyshED: EDCore, partition_function, energies

    
struct AtomicPPGF <: AbstractTimeGF{ComplexF64, false}
    E::Array{Float64, 1}
end

function (P::AtomicPPGF)(t::Number)
    return -im * Diagonal(exp.(-im * t * P.E))
end

function (P::AtomicPPGF)(t1::BranchPoint, t2::BranchPoint)
    Δt = t1.val - t2.val
    return P(Δt)
end

function analytic_atomic_ppgf(ed::EDCore, β::Float64)::Vector{AtomicPPGF}
    Z = partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[i P(β)] = Tr[ρ] = 1)
    P = [ AtomicPPGF(E .+ λ) for E in energies(ed) ]
    return P
end

function analytic_partition_function(Ps::Vector{AtomicPPGF}, β::Float64)
    return sum(Ps, init=0im) do P_s
        im * tr(P_s(-im * β))
    end
end

end # module atomic_ppgf
