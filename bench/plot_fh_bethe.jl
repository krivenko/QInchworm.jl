
import PyPlot as plt
import HDF5; h5 = HDF5


function read_group(group)
    return merge(
        Dict( key => h5.read(group, key) for key in keys(group)),
        Dict( key => h5.read_attribute(group, key) for key in keys(h5.attributes(group)) ) )  
end


filenames = filter( f -> occursin("data_FH_dimer", f), readdir(".", join=true) )
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

# Merge datasets with equal ntau

merged_data = Dict()
for d in data
    key = (d["ntau"], maximum(d["orders"]))
    @show key
    if haskey(merged_data, key)
        for dkey in ["diffs_exa", "diffs_nca", "diffs_oca", "diffs_tca", "N_sampless"]
            merged_data[key][dkey] = vcat(merged_data[key][dkey], d[dkey])
        end
    else
        merged_data[key] = d
    end
end

# sort on N_chunkss

for (key, d) in merged_data
    sort_idx = sortperm(d["N_sampless"])
    for dkey in ["diffs_exa", "diffs_nca", "diffs_oca", "diffs_tca", "N_sampless"]
        d[dkey] = d[dkey][sort_idx]
    end
end

# Get scaling wrt N_tau

diff_keys_order_by_order = Dict(
    1=>("diffs_nca", "diff_0_nca"),
    2=>("diffs_oca", "diff_0_oca"),
    3=>("diffs_tca", "diff_0_tca"),
    4=>("diffs_exa", "diff_0_exa"),
    5=>("diffs_exa", "diff_0_exa"),
    )

diff_keys_cf_exact = Dict(
    1=>("diffs_exa", "diff_0_exa"),
    2=>("diffs_exa", "diff_0_exa"),
    3=>("diffs_exa", "diff_0_exa"),
    4=>("diffs_exa", "diff_0_exa"),
    5=>("diffs_exa", "diff_0_exa"),
    )

#diff_keys = diff_keys_order_by_order
diff_keys = diff_keys_cf_exact

data_keys = sort(collect(keys(merged_data)))
@show data_keys
ntaus = [ key[1] for key in data_keys ]
@show ntaus

# -- Use difference corresponding to order

diffs = Array{Float64}(undef, 0)
rel_diffs = Array{Float64}(undef, 0)
for key in data_keys
    d = merged_data[key]
    max_order = key[2]
    (dkey, d0key) = diff_keys[max_order]

    push!(diffs, d[dkey][end])
    push!(rel_diffs, d[dkey][end] ./ d[d0key])
end

@show rel_diffs
@show diffs

# Plot for all ntau

fig = plt.figure(figsize=(3.24, 6.0*1.25))

gs = fig.add_gridspec(
    nrows=4, ncols=1,
    height_ratios=(1.5, 1.5, 1.5, 1.5),
    left=0.2, right=0.98,
    top=0.99, bottom=0.05,
    wspace=0.3, hspace=0.3)


colors = Dict()
styles = Dict(
    1=> ".",
    2=>"x",
    3=>"+",
    4=>"s",
    5=>"D",
    )


for key in sort(collect(keys(merged_data)))
    
    d = merged_data[key]
    ntau = d["ntau"]
    order_max = maximum(d["orders"])

    if order_max < 4
        idx = 1
    else
        idx = order_max - 2
    end
    
    plt.subplot(gs[idx, 1])

    plt.plot([1e2, 1e4], [1e-1, 1e-3], "-k", lw=0.5, alpha=1.)
    plt.plot([1e1, 1e4], [1e-1, 1e-4], "-k", lw=0.5, alpha=1.)
    
    #N = d["N_chunkss"] .* d["ntau"] .* d["N_per_chunk"]
    N = d["N_sampless"]

    #rel_diffs = d["diffs_nca"] ./ d["diff_0_nca"]
    #diffs = d["diffs_nca"]

    dkey, d0key = diff_keys[order_max]
    rel_diffs = d[dkey] ./ d[d0key]
    diffs = d[dkey]

    style = styles[order_max]
    color = haskey(colors, ntau) ? colors[ntau] : nothing

    if color == nothing
        #label= raw"$N_{\tau}$" * " = $ntau, max(order) = $order_max"
        label= raw"$N_{\tau}$" * " = $ntau"
        
        l = plt.plot([], [], label=label)
        color = l[1].get_color()
        @show color
    end
    
    #plt.loglog(N, rel_diffs, style * "-", color=color,
    plt.loglog(N, diffs, style * "-", color=color,
                   #label=raw"$N_{\tau}$" * " = $ntau, max(order) = $order_max",
                   alpha=0.75, markersize=4, lw=0.5)
    #plt.plot(N[end], rel_diffs[end], "s", color=color, alpha=0.75)
    plt.plot(N[end], diffs[end], "s", color=color, alpha=0.75)

    colors[ntau] = color

    plt.legend(fontsize=7, loc="best", ncol=2, labelspacing=0.1)
    plt.xlabel(raw"$N_{QQMC, tot} / N_{\tau}$", labelpad=0.1)
    #plt.ylabel("Relative Error in ρ")
    plt.ylabel("Error in ρ")
    plt.axis("image")
    plt.grid(true)
    #plt.ylim(bottom=5e-5)
    plt.xlim([1e1, 2e6])
    plt.ylim([1e-5, 1e-1])

end

#for order_max in 1:length(styles)
#    style = styles[order_max]
#    plt.plot([], [], style, color="gray", label="Order = $order_max")
#end


plt.subplot(gs[4, 1])

for order_max in 1:length(styles)
    style = styles[order_max]
    plt.plot([], [], style, color="gray", label="Order = $order_max")
end

#styles = Dict(1=>".-", 2=>"x-", 3=>"+-")
for (order_max, style) in styles
    ntaus_o = [ ntau for (i, ntau) in enumerate(ntaus) if data_keys[i][2] == order_max ]
    rel_diffs_o = [ rel_diff for (i, rel_diff) in enumerate(rel_diffs) if data_keys[i][2] == order_max ]
    #plt.loglog(ntaus_o, rel_diffs_o, "-", color="gray")
    diffs_o = [ diff for (i, diff) in enumerate(diffs) if data_keys[i][2] == order_max ]
    plt.loglog(ntaus_o, diffs_o, "-", color="gray")
end

for i in 1:length(data_keys)
    ntau, order_max = data_keys[i]
    color = colors[ntau]
    #style = Dict(1=>".-", 2=>"x-", 3=>"+-")[order_max]
    style = styles[order_max]
    #plt.loglog(ntau, rel_diffs[i], style, alpha=0.75, color=color)
    plt.loglog(ntau, diffs[i], style, alpha=0.75, color=color)
end

#plt.plot([1e1, 1e2], [1e-1, 1e-3], "-k", lw=3, alpha=0.25)
#plt.plot([1e1, 1e2], [1e-1, 1e-4], "-k", lw=3, alpha=0.25)

plt.xlabel(raw"$N_{\tau}$", labelpad=0.1)
#plt.ylabel("Relative Error in ρ")
plt.ylabel("Error in ρ")
plt.grid(true)
plt.axis("image")
#plt.xlim([2, 4000])
plt.xlim([1e0, 1e4])
plt.legend(fontsize=7, loc="best")

plt.savefig("figure_fh_bethe_convergence.pdf")
plt.show()
