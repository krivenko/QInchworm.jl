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
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Hugo U. R. Strand, Igor Krivenko

using PyPlot; plt = PyPlot
using HDF5; h5 = HDF5

using DataFrames

function read_group(group)
    return merge(
        Dict(key => h5.read(group, key) for key in keys(group)),
        Dict(key => h5.read_attribute(group, key) for key in keys(h5.attributes(group)))
    )
end

filenames = filter( f -> occursin("data_", f), readdir(".", join=true) )

# Load all data files

df = DataFrame(
    beta=Float64[],
    ntau=Int[],
    tau=Vector{Float64}[],
    N_samples=Int[],
    orders=Vector{Int}[],
    P_1=Vector{ComplexF64}[],
    P_2=Vector{ComplexF64}[],
    P_diff_1=Vector{ComplexF64}[],
    P_diff_2=Vector{ComplexF64}[]
)
for filename in filenames
    @show filename
    h5.h5open(filename, "r") do fid
        d = read_group(fid["data"])
        push!(df, d)
    end
end

fig = plt.figure()
groups = groupby(df, ["beta", "orders", "N_samples"])

gs = fig.add_gridspec(nrows=2, ncols=length(groups), wspace=0.3, hspace=0.6)

for (n, gr) in enumerate(groups)

    β = gr[1, "beta"]
    orders = gr[1, "orders"]
    N_samples = gr[1, "N_samples"]

    # Plot differences |P(τ) - P_{diff}(τ)| on τ∈[0; β]
    plt.subplot(gs[1, n])
    for row in eachrow(sort(gr, order("ntau")))
        plt.plot(row["tau"], abs.(row["P_1"] - row["P_diff_1"]),
                 label="\$N_\\tau = $(row["ntau"]), s=1\$")
        plt.xscale("linear")
        plt.yscale("log")
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$|P(\tau) - P_\mathrm{diff}(\tau)|$")
        plt.xlim([0, row["beta"]])
    end
    plt.legend(fontsize=6, loc="best", ncol=2)
    plt.title("\$\\beta=$(β), orders = $(orders), N = $(N_samples)\$",
              fontdict=Dict("fontsize" => 8))

    # Plot scaling of |P(β) - P_{diff}(β)| with N_τ
    plt.subplot(gs[2, n])
    ntaus = [row["ntau"] for row in eachrow(sort(gr, order("ntau")))]
    dP_1_beta = [abs(row["P_1"][end] - row["P_diff_1"][end])
                 for row in eachrow(sort(gr, order("ntau")))]
    dP_2_beta = [abs(row["P_2"][end] - row["P_diff_2"][end])
                 for row in eachrow(sort(gr, order("ntau")))]
    plt.loglog(ntaus, dP_1_beta, label=raw"$s=1$")
    plt.loglog(ntaus, dP_2_beta, label=raw"$s=2$")
    plt.loglog(ntaus, 1 ./ ntaus, ls="--", color="k", label=raw"$1/N_\tau$")
    plt.xlabel(raw"$N_\tau$")
    plt.ylabel(raw"$|P(\beta) - P_\mathrm{diff}(\beta)|$")
    plt.legend(fontsize=6, loc="best")
    plt.title("\$\\beta=$(β), orders = $(orders), N = $(N_samples)\$",
              fontdict=Dict("fontsize" => 8))
end

plt.savefig("figure_diff_inchworm.pdf")
plt.show()
