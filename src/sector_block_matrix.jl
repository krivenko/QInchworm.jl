# QInchworm.jl
#
# Copyright (C) 2021-2026 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Author: Igor Krivenko

"""
Matrix representation of operators acting in a many-body Hilbert space partitioned into
invariant subspaces (sectors) of a Hamiltonian.
"""
module sector_block_matrix

using DocStringExtensions

using LinearAlgebra: norm
import LinearAlgebra
import Statistics

using KeldyshED: EDCore, OperatorExpr, operator_blocks

"""
Complex block matrix stored as a dictionary of non-vanishing blocks.

Each element of the dictionary has the form
`right block index => (left block index, block)`. A block matrix represented by this type is
allowed to have at most one non-vanishing block per column.

Objects of this type support addition/subtraction, matrix multiplication and
multiplication/division by a scalar.
"""
const SectorBlockMatrix = Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}

"""
    $(TYPEDSIGNATURES)

Return the [`SectorBlockMatrix`](@ref) representation of a many-body operator `op`
acting in the Hilbert space of the exact diagonalization problem `ed`.
"""
function operator_to_sector_block_matrix(ed::EDCore, op::OperatorExpr)::SectorBlockMatrix
    sbm = SectorBlockMatrix()
    op_blocks = operator_blocks(ed, op)
    for ((s_f, s_i), mat) in op_blocks
        if haskey(sbm, s_i)
            throw(ArgumentError(
                "Operator $(op) is not representable by a SectorBlockMatrix " *
                "(more than one non-zero blocks per column)"
                )
            )
        end
        sbm[s_i] = (s_f, mat)
    end
    sbm
end

"""
    $(TYPEDSIGNATURES)

Construct a block-diagonal [`SectorBlockMatrix`](@ref), whose block structure is consistent
with the invariant subspace partition of a given exact diagonalization object `ed`.
All matrix elements of the stored blocks are set to zero.
"""
function Base.zeros(::Type{SectorBlockMatrix}, ed::EDCore)::SectorBlockMatrix
    return Dict(s => (s, zeros(ComplexF64, length(sp), length(sp)))
                for (s, sp) in enumerate(ed.subspaces))
end

"""
    $(TYPEDSIGNATURES)

Construct a [`SectorBlockMatrix`](@ref) that shares the list of stored blocks with another
matrix `A` but has all those blocks set to zero.
"""
function Base.zero(A::SectorBlockMatrix)::SectorBlockMatrix
    Z = SectorBlockMatrix()
    for (s_i, (s_f, A_mat)) in A
        Z[s_i] = (s_f, zero(A_mat))
    end
    return Z
end

"""
    $(TYPEDSIGNATURES)

Set elements of all stored blocks of a [`SectorBlockMatrix`](@ref) `A` to `x`.
"""
function Base.fill!(A::SectorBlockMatrix, x)
    for m in A
        fill!(m.second[2], x)
    end
end

function Base.:*(A::SectorBlockMatrix, B::SectorBlockMatrix)::SectorBlockMatrix
    C = SectorBlockMatrix()
    for (s_i, (s, B_mat)) in B
        if haskey(A, s)
            s_f, A_mat = A[s]
            C[s_i] = (s_f, A_mat * B_mat)
        end
    end
    return C
end

function Base.:*(A::Number, B::SectorBlockMatrix)::SectorBlockMatrix
    C = SectorBlockMatrix()
    for (s_i, (s_f, B_mat)) in B
        C[s_i] = (s_f, A * B_mat)
    end
    return C
end

Base.:*(A::SectorBlockMatrix, B::Number) = B * A
Base.:/(A::SectorBlockMatrix, B::Number) = A * (one(B) / B)

function Base.:+(A::SectorBlockMatrix, B::SectorBlockMatrix)::SectorBlockMatrix
    return merge(A, B) do a, b
        @assert a[1] == b[1]
        return (a[1], a[2] + b[2])
    end
end

Base.:-(A::SectorBlockMatrix, B::SectorBlockMatrix) = A + (-1) * B
Base.:-(A::SectorBlockMatrix) = (-1) * A

"""
    $(TYPEDSIGNATURES)

Trace of a [`SectorBlockMatrix`](@ref) `A`.
"""
function LinearAlgebra.tr(A::SectorBlockMatrix)::ComplexF64
    return sum(LinearAlgebra.tr(A_mat) for (s_i, (s_f, A_mat)) in A if s_i == s_f;
               init=zero(ComplexF64))
end

"""
    $(TYPEDSIGNATURES)

`p`-norm of a [`SectorBlockMatrix`](@ref) `A`.
"""
function LinearAlgebra.norm(A::SectorBlockMatrix, p::Real=2)
    isempty(A) && return float(norm(zero(eltype(A))))

    if p == 0
        return sum(a -> norm(a[2], 0), values(A))
    elseif p == Inf
        return maximum(a -> norm(a[2], Inf), values(A))
    elseif p == -Inf
        return minimum(a -> norm(a[2], -Inf), values(A))
    else
        return sum(a -> norm(a[2], p)^p, values(A)) ^ (1/p)
    end
end

"""
    $(TYPEDSIGNATURES)

Inexact equality comparison of two [`SectorBlockMatrix`](@ref) objects `A` and `B`.
Block structures of the objects must agree. `atol` specifies the absolute tolerance for
the single element comparison (zero by default).
"""
function Base.isapprox(A::SectorBlockMatrix, B::SectorBlockMatrix; atol::Real=0)::Bool
    @assert keys(A) == keys(B)
    for k in keys(A)
        @assert A[k][1] == B[k][1]
        !isapprox(A[k][2], B[k][2], norm=mat -> norm(mat, Inf), atol=atol) && return false
    end
    return true
end

"""
    $(TYPEDSIGNATURES)

Compute the sample variance of collection of [`SectorBlockMatrix`](@ref) `itr`.

If `corrected` is `true`, then the sum is scaled with `n-1`, whereas the sum is scaled
with `n` if `corrected` is `false` where `n` is the number of elements in `itr`.
A pre-computed `mean` may be provided.
"""
function Statistics.var(itr::AbstractArray{SectorBlockMatrix};
                        corrected::Bool=true,
                        mean=nothing)
    @assert !isempty(itr)
    v = zero(first(itr))
    for k in keys(first(itr))
        mat_itr = [sbm[k][2] for sbm in itr]
        if mean !== nothing
            @assert mean[k][1] == first(itr)[k][1]
            v[k] = (v[k][1], v[k][2] + Statistics.var(mat_itr, corrected=corrected,
                                                      mean=mean[k][2]))
        else
            v[k] = (v[k][1], v[k][2] + Statistics.var(mat_itr, corrected=corrected))
        end
    end
    return v
end

"""
    $(TYPEDSIGNATURES)

Compute the sample standard deviation of collection `itr`.

If `corrected` is `true`, then the sum is scaled with `n-1`, whereas the sum is scaled
with `n` if `corrected` is `false` where `n` is the number of elements in `itr`.
A pre-computed `mean` may be provided.
"""
function Statistics.std(itr::AbstractArray{SectorBlockMatrix};
                        corrected::Bool=true,
                        mean=nothing)
    return Dict(s_i => (s_f, sqrt.(mat)) for (s_i, (s_f, mat)) in
                Statistics.var(itr, corrected=corrected, mean=mean))
end

end # module sector_block_matrix
