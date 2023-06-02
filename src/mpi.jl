module mpi

using MPI: MPI

using QInchworm.utility: split_count, range_from_chunks_and_idx
using QInchworm.utility: iobuffer_serialize, iobuffer_deserialize

function ismaster(comm::MPI.Comm = MPI.COMM_WORLD)
    return MPI.Comm_rank(comm) == 0
end

function N_skip_and_N_samples_on_rank(N_samples; comm::MPI.Comm = MPI.COMM_WORLD)
    comm_rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    N_split = split_count(N_samples, comm_size)
    N_skip = sum(N_split[1:comm_rank])
    N_samples_on_rank = N_split[comm_rank+1]
    return N_skip, N_samples_on_rank
end

function rank_sub_range(n; comm::MPI.Comm = MPI.COMM_WORLD)
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm) # zero based indexing
    chunks = split_count(n, comm_size)
    return range_from_chunks_and_idx(chunks, comm_rank + 1)
end

function all_gather(subvec::Vector{T}; comm::MPI.Comm = MPI.COMM_WORLD) where T
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

end
