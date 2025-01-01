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
Exact atomic pseudo-particle Green's function module enabling exact evaluation of the atomic
propagator ``P_0(z)`` by evaluating the exponential ``P_0(z) = -i e^{-iz \\hat H_{loc}}``.

# Exports
$(EXPORTS)
"""
module exact_atomic_ppgf

using DocStringExtensions

using LinearAlgebra: Diagonal, tr

using Keldysh: BranchPoint, AbstractTimeGF
using KeldyshED; ked = KeldyshED;

import Keldysh: interpolate!
import QInchworm.ppgf: partition_function, atomic_ppgf, density_matrix

export ExactAtomicPPGF, partition_function, atomic_ppgf, density_matrix, interpolate!

"""
$(TYPEDEF)

Exact atomic pseudo-particle Green's function type.

# Fields
$(TYPEDFIELDS)
"""
struct ExactAtomicPPGF <: AbstractTimeGF{ComplexF64, false}
    "Inverse temperature"
    β::Float64
    "Eigenvalues of the atomic Hamiltonian"
    E::Array{Float64, 1}
end

"""
    $(TYPEDSIGNATURES)

Evaluate atomic propagator at complex contour time `z`.

# Parameters
- `z`: scalar time.

# Returns
- Value of atomic pseudo-particle propagator ``P_0(z)`` as a diagonal matrix
  `Diagonal`.

"""
function (P::ExactAtomicPPGF)(z::Number)
    return -im * Diagonal(exp.(-im * z * P.E))
end

"""
    $(TYPEDSIGNATURES)

Evaluate atomic propagator at the difference between imaginary time branch points.

# Parameters
- `z1`: first branch point.
- `z2`: second branch point.

# Returns
- Value of atomic pseudo-particle propagator ``P_0(z_1 - z_2)`` as a diagonal matrix
  `Diagonal`.

"""
function (P::ExactAtomicPPGF)(z1::BranchPoint, z2::BranchPoint)
    Δz = z1.val - z2.val
    return P(Δz)
end

"""
    $(TYPEDSIGNATURES)

In-place evaluation of the atomic propagator at the difference between imaginary time branch
points.

# Parameters
- `x`: Matrix to store the value of the atomic pseudo-particle propagator in.
- `P_0`: Atomic pseudo-particle propagator.
- `z1`: first branch point.
- `z2`: second branch point.

# Returns
- Value of atomic pseudo-particle propagator ``P_0(z_1 - z_2)`` as a diagonal matrix
  `Diagonal`.

"""
@inline function interpolate!(x::Matrix{ComplexF64},
                              P_0::ExactAtomicPPGF,
                              z1::BranchPoint,
                              z2::BranchPoint)
    Δz = z1.val - z2.val
    fill!(x, 0.0)
    for i in eachindex(P_0.E)
        x[i, i] = -im * exp(-im * Δz * P_0.E[i])
    end
end

"""
    $(TYPEDSIGNATURES)

Construct the exact atomic pseudo-particle Green's function.

# Parameters
- `β`: Inverse temperature.
- `ed`: Exact diagonalization structure describing the atomic problem.

"""
function atomic_ppgf(β::Float64, ed::ked.EDCore)::Vector{ExactAtomicPPGF}
    Z = ked.partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[i P(β)] = Tr[ρ] = 1)
    P = [ ExactAtomicPPGF(β, E .+ λ) for E in energies(ed) ]
    return P
end

"""
    $(TYPEDSIGNATURES)

Extract the partition function ``Z = \\mathrm{Tr}[i P_0(-i\\beta, 0)]`` from a un-normalized
pseudo-particle Green's function `P_0`.
"""
function partition_function(P_0::Vector{ExactAtomicPPGF})::ComplexF64
    return sum(P_0, init=0im) do P_s
        im * tr(P_s(-im * P_s.β))
    end
end

"""
    $(TYPEDSIGNATURES)

Extract the equilibrium density matrix ``\\rho = i P(-i\\beta, 0)`` from a normalized
pseudo-particle Green's function `P`. The density matrix is block-diagonal and is returned
as a vector of blocks.
"""
function density_matrix(P::Vector{ExactAtomicPPGF})::Vector{Matrix{ComplexF64}}
    @assert !isempty(P)
    return [1im * P_s(-im * P_s.β) for P_s in P]
end

end # module exact_atomic_ppgf
