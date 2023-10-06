
using PyPlot; plt = PyPlot
using HDF5; h5 = HDF5

function read_group(group)
    return merge(
        Dict( key => h5.read(group, key) for key in keys(group)),
        Dict( key => h5.read_attribute(group, key) for key in keys(h5.attributes(group)) ) )
end


filenames = filter( f -> occursin("data_bethe_ntau", f), readdir(".", join=true) )
@show filenames

# load all data files

data = []
for filename in filenames
    @show filename
    fid = h5.h5open(filename, "r")
    g = fid["data"]
    d = read_group(g)
    h5.close(fid)
    push!(data, d)
end

# Merge datasets with equal nτ

merged_data = Dict()
for d in data
    key = (d["ntau"], maximum(d["orders"]))
    @show key
    if haskey(merged_data, key)
        for dkey in ["diffs", "N_sampless"]
            merged_data[key][dkey] = vcat(merged_data[key][dkey], d[dkey])
        end
    else
        merged_data[key] = d
    end
end

# sort on N_chunkss

for (key, d) in merged_data
    sort_idx = sortperm(d["N_sampless"])
    for dkey in ["diffs", "N_sampless"]
        d[dkey] = d[dkey][sort_idx]
    end
end

# Get scaling wrt N_tau

data_keys = sort(collect(keys(merged_data)))
@show data_keys
nτs = [ key[1] for key in data_keys ]
@show nτs
rel_diffs = [ d["diffs"][end] ./ d["diff_0"] for d in [ merged_data[key] for key in data_keys ] ]
diffs = [ d["diffs"][end] for d in [ merged_data[key] for key in data_keys ] ]
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
    5=>"x",
    )

for key in sort(collect(keys(merged_data)))
    d = merged_data[key]
    nτ = d["ntau"]
    order_max = maximum(d["orders"])

    N = d["N_sampless"]
    rel_diffs = d["diffs"] ./ d["diff_0"]
    diffs = d["diffs"]

    style = styles[order_max]
    color = haskey(colors, nτ) ? colors[nτ] : nothing

    if isnothing(color)
        #label= raw"$N_{\tau}$" * " = $nτ, max(order) = $order_max"
        label= raw"$N_{\tau}$" * " = $nτ"

        l = plt.plot([], [], label=label)
        color = l[1].get_color()
        @show color
    end

    plt.loglog(N, diffs, style * "-", color=color,
                   #label=raw"$N_{\tau}$" * " = $nτ, max(order) = $order_max",
                   alpha=0.75, markersize=4, lw=0.5)
    plt.plot(N[end], diffs[end], "s", color=color, alpha=0.75)

    colors[nτ] = color
end

#for order_max in 1:length(styles)
#    style = styles[order_max]
#    plt.plot([], [], style, color="gray", label="Order = $order_max")
#end

plt.legend(fontsize=7, loc="best", labelspacing=0.1)
plt.xlabel(raw"$N_{QQMC, tot} / N_{\tau}$")
plt.ylabel("Relative Error in ρ")
plt.axis("image")
plt.grid(true)
#plt.ylim(bottom=5e-5)

plt.subplot(gs[2, 1])

#for order_max in 1:length(styles)
#for (order_max, style) in styles
for order_max in sort(collect(keys(styles)))
    style = styles[order_max]
    plt.plot([], [], style, color="gray", label="Order = $order_max")
end

#styles = Dict(1=>".-", 2=>"x-", 3=>"+-")
for (order_max, style) in styles
    nτs_o = [ nτ for (i, nτ) in enumerate(nτs) if data_keys[i][2] == order_max ]
    #rel_diffs_o = [ rel_diff for (i, rel_diff) in enumerate(rel_diffs) if data_keys[i][2] == order_max ]
    diffs_o = [ diff for (i, diff) in enumerate(diffs) if data_keys[i][2] == order_max ]
    plt.loglog(nτs_o, diffs_o, "-", color="gray")
end

for i in eachindex(data_keys)
    nτ, order_max = data_keys[i]
    color = colors[nτ]
    #style = Dict(1=>".-", 2=>"x-", 3=>"+-")[order_max]
    style = styles[order_max]
    plt.loglog(nτ, diffs[i], style, alpha=0.75, color=color)
end

#plt.plot([1e1, 1e2], [1e-1, 1e-3], "-k", lw=3, alpha=0.25)
#plt.plot([1e1, 1e2], [1e-1, 1e-4], "-k", lw=3, alpha=0.25)

plt.xlabel(raw"$N_{\tau}$")
plt.ylabel("Relative Error in ρ")
plt.grid(true)
plt.axis("image")
#plt.xlim([2, 4000])
plt.xlim([1e1, 1e4])
plt.legend(fontsize=7, loc="best")

plt.savefig("figure_bethe_convergence_order.pdf")
plt.show()
