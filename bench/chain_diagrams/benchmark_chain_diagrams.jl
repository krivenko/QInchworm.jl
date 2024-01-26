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
# Authors: Igor Krivenko

using MPI; MPI.Init()
using HDF5; h5 = HDF5

using ArgParse
using MD5

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.sector_block_matrix: SectorBlockMatrix
using QInchworm.utility: ph_conj
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.diagrammatics: Topology, is_k_connected
using QInchworm.topology_eval: TopologyEvaluator, IdentityNode
using QInchworm.scrambled_sobol: ScrambledSobolSeq, skip!
using QInchworm.qmc_integrate: contour_integral, RootTransform

using QInchworm.mpi: ismaster, rank_sub_range, all_reduce!

raw"""
Use qMC to compute diagrammatic contributions to the PPGF derived from a single chain
topology.

The chain topology of order `n` has the following form,
         n     n-1    n-2          3      2      1
       _____  _____  _____       _____  _____  _____
      /     \/     \/     \ ... /     \/     \/     \
     /      /\     /\     /     \     /\     /\      \
β ==*======*==*===*==*===*=======*===*==*===*==*======*== 0

with the total number of pair interaction arcs equal to `n`.

For the single band Anderson model, there are exactly 8 contributing configurations stemming
from the chain topology *regardless of its order*,

- ... c^†(dn) c(dn) c^†(up) c(up) c^†(dn) c^†(up)
- ... c(dn) c^†(dn) c(up) c^†(up) c(dn) c(up)
- ... c^†(dn) c(dn) c(up) c^†(up) c^†(dn) c(up)
- ... c(dn) c^†(dn) c^†(up) c(up) c(dn) c^†(up)

+ 4 spin conjugates of these configurations.
"""
function compute_chain_diagram(order, nτ, N_samples)

    β = 10.0
    μ = 2.0
    U = 4.0
    ϵ = 0.2
    V = 1.0

    # ED solution

    H_imp = U * op.n("up") * op.n("dn") - μ * (op.n("up") + op.n("dn"))

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    soi = ked.Hilbert.SetOfIndices([["up"], ["dn"]])
    ed = ked.EDCore(H_imp, soi)

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ), grid)

    # Pseudo Particle Strong Coupling Expansion

    ips = InteractionPair{kd.ImaginaryTimeGF{ComplexF64, true}}[]
    for s in ["up", "dn"]
        push!(ips, InteractionPair(op.c_dag(s), op.c(s), Δ))
        push!(ips, InteractionPair(op.c(s), op.c_dag(s), ph_conj(Δ)))
    end
    expansion = Expansion(ed, grid, ips)

    # Prepare topology evaluator

    τ_grid = grid[kd.imaginary_branch]
    t_i, t_f = τ_grid[1].bpoint, τ_grid[end].bpoint

    n_0 = IdentityNode(t_i)
    n_f = IdentityNode(t_f)

    teval = TopologyEvaluator(expansion, order, false, Dict(1 => n_0, 2order + 2 => n_f))

    # Construct the chain topology

    @assert order > 0
    if order == 1
        pairs = [1 => 2]
    else
        pairs = vcat([1 => 3],
                     [o => o + 3 for o in 2:2:2order-4],
                     [2order-2 => 2order])
    end

    chain_topology = Topology(pairs)
    @assert is_k_connected(chain_topology, 1)
    ismaster() && println("Chain topology of order $(order): $(chain_topology)")

    # Do the integration

    trans = RootTransform(2order, contour, t_i, t_f)

    N_range = rank_sub_range(N_samples)
    rank_weight = length(N_range) / N_samples

    seq = ScrambledSobolSeq(2order)
    skip!(seq, first(N_range) - 1, exact=true)
    res::SectorBlockMatrix = rank_weight * contour_integral(
        t -> teval(chain_topology, t),
        contour,
        trans,
        init = zeros(SectorBlockMatrix, expansion.ed),
        seq = seq,
        N = length(N_range)
    )
    all_reduce!(res, +)

    result = [res[i][2][1, 1] for i in 1:length(res)]
    if ismaster()
        @show N_samples
        @show result
    end
    return result
end

s = ArgParseSettings("Computation of 'chain' type diagrams for the Anderson model")
@add_arg_table s begin
    "order"
        help = "Order of the chain diagram to be computed"
        arg_type = Int
    "ntau"
        help = "Number of imaginary time slices in interpolated functions"
        arg_type = Int
    "N_samples_min_log2"
        help = "Minimum number of qMC samples to be taken in log2 scale"
        arg_type = Int
    "N_samples_max_log2"
        help = "Maximum number of qMC samples to be taken in log2 scale"
        arg_type = Int
end

parsed_args = parse_args(ARGS, s)

order = parsed_args["order"]
nτ = parsed_args["ntau"]
N_samples_min_log2 = parsed_args["N_samples_min_log2"]
N_samples_max_log2 = parsed_args["N_samples_max_log2"]
N_samples_list = 2 .^ (N_samples_min_log2:N_samples_max_log2)

results = [compute_chain_diagram(order, nτ, N_samples) for N_samples in N_samples_list]

if ismaster()
    id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(results...))))
    filename = "chain_diagrams_ntau_$(nτ)_order_$(order)_md5_$(id).h5"
    @show filename

    h5.h5open(filename, "w") do fid
        g = h5.create_group(fid, "data")

        h5.attributes(g)["ntau"] = nτ
        h5.attributes(g)["N_samples"] = collect(N_samples_list)

        for (order, r) in enumerate(results)
            g[string(order)] = results[order]
        end
    end
end
