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

using MD5
using ArgParse

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.utility: ph_conj
using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, diff_inchworm!
using QInchworm.mpi: ismaster

# Single state pseudo particle expansion

β = 10.
μ = +0.1 # Chemical potential
ϵ = +0.1 # Bath energy level
V = -0.1 # Hybridization

H = - μ * op.n("0")
soi = ked.Hilbert.SetOfIndices([["0"]])
ed = ked.EDCore(H, soi)
ρ = ked.density_matrix(ed, β)

function run_inchworm(nτ, orders, orders_bare, N_samples)
    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ), grid)

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), ph_conj(Δ))
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    inchworm!(expansion, grid, orders, orders_bare, N_samples)

    return expansion.P
end

function run_diff_inchworm(nτ, orders, N_samples)
    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    Δ = V^2 * kd.ImaginaryTimeGF(kd.DeltaDOS(ϵ), grid)

    ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
    ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), ph_conj(Δ))
    expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

    diff_inchworm!(expansion, grid, orders, N_samples)

    return expansion.P
end

s = ArgParseSettings()
@add_arg_table s begin
    "order"
        help = "Highest expansion order to account for"
        arg_type = Int
    "ntau"
        help = "Number of imaginary time slices"
        arg_type = Int
    "N_samples"
        help = "Number of qMC samples to be taken"
        arg_type = Int
end

parsed_args = parse_args(ARGS, s)

order = parsed_args["order"]
nτ = parsed_args["ntau"]
N_samples = parsed_args["N_samples"]

if ismaster()
    println("order $(order) nτ $(nτ) N_samples $(N_samples)")
end

orders = 0:order

P = run_inchworm(nτ, orders, orders, N_samples)
P_diff = run_diff_inchworm(nτ, orders, N_samples)

if ismaster()
    id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(P[1].mat.data...))))
    filename = "data_order_$(orders)_ntau_$(nτ)_N_samples_$(N_samples)_md5_$(id).h5"
    @show filename

    h5.h5open(filename, "w") do fid
        grp = h5.create_group(fid, "data")

        h5.attributes(grp)["beta"] = β
        h5.attributes(grp)["ntau"] = nτ
        h5.attributes(grp)["N_samples"] = N_samples

        grp["orders"] = collect(orders)

        grp["tau"] = collect(kd.imagtimes(P[1].grid))

        grp["P_1"] = P[1].mat.data[1, 1, :]
        grp["P_2"] = P[2].mat.data[1, 1, :]
        grp["P_diff_1"] = P_diff[1].mat.data[1, 1, :]
        grp["P_diff_2"] = P_diff[2].mat.data[1, 1, :]
    end
end
