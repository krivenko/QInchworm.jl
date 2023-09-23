using DocStringExtensions

using LinearAlgebra: norm
import LinearAlgebra

using KeldyshED

"""
Complex block matrix stored as a dictionary of non-vanishing blocks.

An element of the dictionary has the form
right block index => (left block index, block).
"""
const SectorBlockMatrix = Dict{Int64, Tuple{Int64, Matrix{ComplexF64}}}
const SectorBlockMatrixReal = Dict{Int64, Tuple{Int64, Matrix{Float64}}}
GenericSectorBlockMatrix = Union{SectorBlockMatrix, SectorBlockMatrixReal}

"""
$(TYPEDSIGNATURES)

Returns the [`SectorBlockMatrix`](@ref) representation of the many-body operator.
"""
function operator_to_sector_block_matrix(ed::KeldyshED.EDCore,
                                         op::KeldyshED.OperatorExpr)::SectorBlockMatrix
    sbm = SectorBlockMatrix()
    op_blocks = KeldyshED.operator_blocks(ed, op)
    for ((s_f, s_i), mat) in op_blocks
        sbm[s_i] = (s_f, mat)
    end
    sbm
end

"""
$(TYPEDSIGNATURES)

Construct a block-diagonal complex matrix, whose block structure is consistent
with the invariant subspace partition of a given KeldyshED.EDCore object.
All matrix elements of the stored blocks are set to zero.
"""
function Base.zeros(::Type{SectorBlockMatrix}, ed::KeldyshED.EDCore)
    Dict(s => (s, zeros(ComplexF64, length(sp), length(sp))) for (s, sp) in enumerate(ed.subspaces))
end

function Base.zeros(::Type{SectorBlockMatrixReal}, ed::KeldyshED.EDCore)
    Dict(s => (s, zeros(Float64, length(sp), length(sp))) for (s, sp) in enumerate(ed.subspaces))
end

function Base.zero(A::GenericSectorBlockMatrix)
    Z = typeof(A)()
    for (s_i, (s_f, A_mat)) in A
        Z[s_i] = (s_f, zero(A_mat))
    end
    return Z
end

function Base.fill!(A::GenericSectorBlockMatrix, x)
    for m in A
        fill!(m.second[2], x)
    end
end

function Base.:*(A::GenericSectorBlockMatrix, B::GenericSectorBlockMatrix)
    C = typeof(A)()
    for (s_i, (s, B_mat)) in B
        if haskey(A, s)
            s_f, A_mat = A[s]
            C[s_i] = (s_f, A_mat * B_mat)
        end
    end
    return C
end

function Base.:*(A::Number, B::GenericSectorBlockMatrix)
    C = typeof(B)()
    for (s_i, (s_f, B_mat)) in B
        C[s_i] = (s_f, A * B_mat)
    end
    return C
end

Base.:*(A::GenericSectorBlockMatrix, B::Number) = B * A

function Base.:+(A::GenericSectorBlockMatrix, B::GenericSectorBlockMatrix)
    return merge(A, B) do a, b
        @assert a[1] == b[1]
        return (a[1], a[2] + b[2])
    end
end

Base.:-(A::GenericSectorBlockMatrix, B::GenericSectorBlockMatrix) = A + (-1) * B

function LinearAlgebra.tr(A::SectorBlockMatrix)
    sum(LinearAlgebra.tr(A_mat) for (s_i, (s_f, A_mat)) in A if s_i == s_f;
        init=ComplexF64(0))
end

function Base.isapprox(A::GenericSectorBlockMatrix, B::GenericSectorBlockMatrix; atol::Real=0)
    @assert keys(A) == keys(B)
    for k in keys(A)
        @assert A[k][1] == B[k][1]
        !isapprox(A[k][2], B[k][2], norm = mat -> norm(mat, Inf), atol=atol) && return false
    end
    return true
end

function Base.map(f, A::GenericSectorBlockMatrix)
    B = typeof(A)()
    for (s_i, (s_f, A_mat)) in A
        B[s_i] = (s_f, f.(A_mat))
    end
    return B
end

function Base.abs(A::GenericSectorBlockMatrix)
    return real(map(abs, A))
end

function Base.maximum(A::SectorBlockMatrixReal)
    return maximum([ maximum(mat) for (s_i, (s_f, mat)) in A ])
end

function Base.imag(A::SectorBlockMatrix)
    A_im = SectorBlockMatrixReal()
    for (s_i, (s_f, A_mat)) in A
        A_im[s_i] = (s_f, imag(A_mat))
    end
    return A_im
end

function Base.real(A::SectorBlockMatrix)
    A_im = SectorBlockMatrixReal()
    for (s_i, (s_f, A_mat)) in A
        A_im[s_i] = (s_f, real(A_mat))
    end
    return A_im
end
