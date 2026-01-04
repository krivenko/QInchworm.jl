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
# Authors: Hugo U. R. Strand, Igor Krivenko

using PyPlot; plt = PyPlot
using HDF5; h5 = HDF5

function read_group(group)
    return merge(
        Dict(key => h5.read(group, key) for key in keys(group)),
        Dict(key => h5.read_attribute(group, key) for key in keys(h5.attributes(group)))
    )
end

filenames = filter(f -> occursin("data_ntau", f), readdir(".", join=true))

# Load all data files

data = []
for filename in filenames
    @show filename
    h5.h5open(filename, "r") do fid
        d = read_group(fid["data"])
        push!(data, d)
    end
end

# Merge datasets with equal nτ

merged_data = Dict()
for d in data
    key = (d["ntau"], maximum(d["orders"]))
    if haskey(merged_data, key)
        for dkey in ["diffs", "N_sampless"]
            merged_data[key][dkey] = vcat(merged_data[key][dkey], d[dkey])
        end
    else
        merged_data[key] = d
    end
end

# Sort on N_sampless

for (key, d) in merged_data
    sort_idx = sortperm(d["N_sampless"])
    for dkey in ["diffs", "N_sampless"]
        d[dkey] = d[dkey][sort_idx]
    end
end

# Get scaling w.r.t. nτ

data_keys = sort(collect(keys(merged_data)))
nτs = [key[1] for key in data_keys]
@show nτs
rel_diffs = [d["diffs"][end] ./ d["diff_0"]
             for d in [merged_data[key] for key in data_keys]]
@show rel_diffs

# Plot for all nτ

fig = plt.figure(figsize=(3.24, 6.))
gs = fig.add_gridspec(
    nrows=2, ncols=1,
    height_ratios=(1.5, 1),
    left=0.2, right=0.99,
    top=0.99, bottom=0.10,
    wspace=0.3, hspace=0.3)

plt.subplot(gs[1, 1])

plt.plot([1e2, 1e4], [1e-1, 1e-3], "-k", lw=3, alpha=0.25)
plt.plot([1e1, 1e4], [1e-1, 1e-4], "-k", lw=3, alpha=0.25)

colors = Dict()
styles = Dict(
    1=> ".",
    #2=>"x",
    3=>"+",
    )

for key in sort(collect(keys(merged_data)))
    d = merged_data[key]
    nτ = d["ntau"]
    order_max = maximum(d["orders"])

    N = d["N_sampless"]
    local rel_diffs = d["diffs"] ./ d["diff_0"]

    style = styles[order_max]
    color = haskey(colors, nτ) ? colors[nτ] : nothing

    if isnothing(color)
        #label= raw"$N_{\tau}$" * " = $nτ, max(order) = $order_max"
        label= raw"$N_{\tau}$" * " = $nτ"
        l = plt.plot([], [], label=label)
        color = l[1].get_color()
    end

    plt.loglog(N, rel_diffs, style * "-", color=color,
               #label=raw"$N_{\tau}$" * " = $nτ, max(order) = $order_max",
               alpha=0.75, markersize=4, lw=0.5)
    plt.plot(N[end], rel_diffs[end], "s", color=color, alpha=0.75)

    colors[nτ] = color
end

#for order_max in 1:length(styles)
#    style = styles[order_max]
#    plt.plot([], [], style, color="gray", label="Order = $order_max")
#end

plt.legend(fontsize=7, loc="best", labelspacing=0.1)
plt.xlabel(raw"$N_{QQMC, tot} / N_{\tau}$")
plt.ylabel(raw"Relative Error in $\rho$")
plt.axis("image")
plt.grid(true)

plt.subplot(gs[2, 1])

for order_max in sort(collect(keys(styles)))
    style = styles[order_max]
    plt.plot([], [], style, color="gray", label="Order = $order_max")
end

for (order_max, style) in styles
    nτs_o = [nτ for (i, nτ) in enumerate(nτs) if data_keys[i][2] == order_max]
    rel_diffs_o = [rel_diff for (i, rel_diff) in enumerate(rel_diffs)
                   if data_keys[i][2] == order_max]
    plt.loglog(nτs_o, rel_diffs_o, "-", color="gray")
end

for i in eachindex(data_keys)
    nτ, order_max = data_keys[i]
    color = colors[nτ]
    #style = Dict(1=>".-", 2=>"x-", 3=>"+-")[order_max]
    style = styles[order_max]
    plt.loglog(nτ, rel_diffs[i], style, alpha=0.75, color=color)
end

plt.xlabel(raw"$N_{\tau}$")
plt.ylabel(raw"Relative Error in $\rho$")
plt.grid(true)
plt.axis("image")
plt.xlim([1e1, 1e4])
plt.legend(fontsize=7, loc="best")

plt.savefig("figure_dimer_convergence_order.pdf")
plt.show()
