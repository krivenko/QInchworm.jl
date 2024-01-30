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
# Authors: Igor Krivenko, Hugo U. R. Strand

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

using QInchworm.keldysh_dlr: DLRImaginaryTimeGrid, DLRImaginaryTimeGF, ph_conj
import Lehmann; le = Lehmann

using QInchworm.analytic_ppgf: ScalarAnalyticGF, analytic_gf

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

"""
function compute_chain_diagram(order, ϵ, V, β, N_samples)

    # ED solution

    H_imp = 0 * op.n(1)

    # Impurity problem

    nτ = 2
    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    soi = ked.Hilbert.SetOfIndices([[1],])
    ed = ked.EDCore(H_imp, soi)

    # Hybridization propagator

    #Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ), grid)
    #ips = InteractionPair{kd.ImaginaryTimeGF{ComplexF64, true}}[]

    #dlr = le.DLRGrid(Euv=8., β=β, isFermi=true, rtol=1e-12, rebuild=true, verbose=ismaster())
    #dlr_grid = DLRImaginaryTimeGrid(contour, dlr)
    #Δ = DLRImaginaryTimeGF(kd.DeltaDOS(ϵ), dlr_grid)
    #Δ.mat.data[:] = Δ.mat.data .* V^2
    #ips = InteractionPair{DLRImaginaryTimeGF{ComplexF64, true}}[]
    
    Δ = analytic_gf(β, ϵ)
    ips = InteractionPair{ScalarAnalyticGF{ComplexF64, true}}[]

    push!(ips, InteractionPair(op.n(1), op.n(1), Δ))

    # Pseudo Particle Strong Coupling Expansion

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

    result = (-1)^order * imag(res[2][2][1, 1])
    
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
    "epsilon"
        help = "Bath energy level"
        arg_type = Float64
    "V"
        help = "Bath hybridization"
        arg_type = Float64
    "beta"
        help = "Inverse temperature"
        arg_type = Float64
    "N_samples_min_log2"
        help = "Minimum number of qMC samples to be taken in log2 scale"
        arg_type = Int
    "N_samples_max_log2"
        help = "Maximum number of qMC samples to be taken in log2 scale"
        arg_type = Int
end

parsed_args = parse_args(ARGS, s)

order = parsed_args["order"]
ϵ = parsed_args["epsilon"]
V = parsed_args["V"]
β = parsed_args["beta"]
N_samples_min_log2 = parsed_args["N_samples_min_log2"]
N_samples_max_log2 = parsed_args["N_samples_max_log2"]
N_samples_list = 2 .^ (N_samples_min_log2:N_samples_max_log2)

results = [compute_chain_diagram(order, ϵ, V, β, N_samples) for N_samples in N_samples_list]

if ismaster()
    id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(results...))))
    filename = "data_chain_diagrams_order_$(order)_eps_$(eps)_V_$(V)_beta_$(β)_md5_$(id).h5"
    @show filename

    h5.h5open(filename, "w") do fid
        g = h5.create_group(fid, "data")

        h5.attributes(g)["order"] = order

        h5.attributes(g)["epsilon"] = ϵ
        h5.attributes(g)["V"] = V
        h5.attributes(g)["beta"] = β
        
        g["N_samples_list"] = N_samples_list
        g["results"] = results
    end
end
