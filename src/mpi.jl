# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/.
#
# Author: Igor Krivenko

module mpi

using MPI: MPI

using QInchworm: SectorBlockMatrix
using QInchworm.utility: split_count, range_from_chunks_and_idx
using QInchworm.utility: iobuffer_serialize, iobuffer_deserialize

function ismaster(comm::MPI.Comm = MPI.COMM_WORLD)::Bool
    return MPI.Comm_rank(comm) == 0
end

function N_skip_and_N_samples_on_rank(N_samples::Integer; comm::MPI.Comm = MPI.COMM_WORLD)
    comm_rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    N_split = split_count(N_samples, comm_size)
    N_skip = sum(N_split[1:comm_rank])
    N_samples_on_rank = N_split[comm_rank+1]
    return N_skip, N_samples_on_rank
end

function rank_sub_range(n::Integer; comm::MPI.Comm = MPI.COMM_WORLD)
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm) # zero based indexing
    chunks = split_count(n, comm_size)
    return range_from_chunks_and_idx(chunks, comm_rank + 1)
end

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

function all_reduce!(sbm::SectorBlockMatrix, op; comm::MPI.Comm = MPI.COMM_WORLD)
    for (s_i, (s_f, mat)) in sbm
        MPI.Allreduce!(mat, op, comm)
    end
end

end # module mpi
