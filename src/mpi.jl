# QInchworm.jl
#
# Copyright (C) 2021-2024 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
MPI-related utility functions.
"""
module mpi

using DocStringExtensions

using MPI: MPI

using QInchworm.sector_block_matrix: SectorBlockMatrix
using QInchworm.utility: split_count, range_from_chunks_and_idx
using QInchworm.utility: iobuffer_serialize, iobuffer_deserialize

"""
    $(TYPEDSIGNATURES)

Check whether the calling process has the rank 0 within the group associated with the
communicator `comm`.
"""
function ismaster(comm::MPI.Comm = MPI.COMM_WORLD)::Bool
    return MPI.Comm_rank(comm) == 0
end

"""
    $(TYPEDSIGNATURES)

Split the range `1:N` between MPI processes in the communicator `comm` as evenly as
possible and return the sub-range 'owned' by the calling process.
"""
function rank_sub_range(N::Integer; comm::MPI.Comm = MPI.COMM_WORLD)::UnitRange{Int}
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm) # zero based indexing
    chunks = split_count(N, comm_size)
    return range_from_chunks_and_idx(chunks, comm_rank + 1)
end

"""
    $(TYPEDSIGNATURES)

Perform the MPI collective operation `Allgather` for vectors of elements of a generic type
`T`.

# Parameters
- `subvec`: Subvector to be gathered.
- `comm`:   MPI communicator.

# Returns
The gathered vector of the same element type `T`.
"""
function all_gather(subvec::Vector{T}; comm::MPI.Comm = MPI.COMM_WORLD)::Vector{T} where T
    data_raw = iobuffer_serialize(subvec)
    data_size = length(data_raw)

    size = [data_size]
    sizes = MPI.Allgather(size, comm)

    output = zeros(UInt8, sum(sizes))
    output_vbuf = MPI.VBuffer(output, sizes)

    MPI.Allgatherv!(data_raw, output_vbuf, comm)

    out_vec = T[]
    for i = 1:length(sizes)
        r = range_from_chunks_and_idx(sizes, i)
        subvec_i = iobuffer_deserialize(output[r])
        append!(out_vec, subvec_i)
    end
    return out_vec
end

"""
    $(TYPEDSIGNATURES)

Perform the in-place MPI collective operation `Allreduce` for a [sector block matrix](
@ref SectorBlockMatrix).

# Parameters
- `sbm`:  Sector block matrices to be reduced.
- `op`:   Reduction operation.
- `comm`: MPI communicator.

# Returns
The gathered vector of the same element type `T`.
"""
function all_reduce!(sbm::SectorBlockMatrix, op; comm::MPI.Comm = MPI.COMM_WORLD)
    # Reduce number of MPI.Allreduce! calls to one by packing all sbm sectors in one vector,
    # allreducing the vector, and then unpacking the vector.

    sector_sizes = [ prod(size(mat)) for (s_i, (s_f, mat)) in sbm ]
    N = sum(sector_sizes)
    v = Vector{ComplexF64}(undef, N)

    e = 0
    for (n, (s_i, (s_f, mat))) in zip(sector_sizes, sbm)
        s = e + 1
        e += n
        v[s:e] .= vec(mat)
    end

    MPI.Allreduce!(v, op, comm)

    e = 0
    for (n, (s_i, (s_f, mat))) in zip(sector_sizes, sbm)
        s = e + 1
        e += n
        vec(mat) .= v[s:e]
    end
end

end # module mpi
