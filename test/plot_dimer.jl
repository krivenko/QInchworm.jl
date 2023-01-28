
import PyPlot as plt
import HDF5; h5 = HDF5


function read_group(group)
    return merge(
        Dict( key => h5.read(group, key) for key in keys(group)),
        Dict( key => h5.read_attribute(group, key) for key in keys(h5.attributes(group)) ) )  
end


filenames = filter( f -> occursin("data_ntau", f), readdir(".", join=true) )
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
    key = d["ntau"]
    if haskey(merged_data, key)
        for dkey in ["diffs", "N_chunkss"]
            merged_data[key][dkey] = vcat(merged_data[key][dkey], d[dkey])
        end
    else
        merged_data[key] = d
    end
end

# sort on N_chunkss

for (key, d) in merged_data
    sort_idx = sortperm(d["N_chunkss"])
    for dkey in ["diffs", "N_chunkss"]
        d[dkey] = d[dkey][sort_idx]
    end
end

# Get scaling wrt N_tau

ntaus = sort(collect(keys(merged_data)))
rel_diffs = [ d["diffs"][end] ./ d["diff_0"] for d in [ merged_data[ntau] for ntau in ntaus ] ]

# Plot for all ntau

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

for key in sort(collect(keys(merged_data)))
    d = merged_data[key]
    ntau = d["ntau"]
    #N = d["N_chunkss"] .* d["ntau"] .* d["N_per_chunk"]
    N = d["N_chunkss"] .* d["N_per_chunk"]
    rel_diffs = d["diffs"] ./ d["diff_0"]
    l = plt.loglog(N, rel_diffs, ".-",
                   label=raw"$N_{\tau}$" * " = $ntau", alpha=0.75)
    color = l[1].get_color()
    @show color
    plt.plot(N[end], rel_diffs[end], "s", color=color, alpha=0.75)
end

plt.legend(fontsize=7, loc="best")
plt.xlabel(raw"$N_{QQMC, tot} / N_{\tau}$")
plt.ylabel("Relative Error in ρ")
plt.axis("image")
plt.grid(true)
#plt.ylim(bottom=5e-5)
plt.ylim(bottom=1e-5)

plt.subplot(gs[2, 1])
plt.loglog(ntaus, rel_diffs, "-", color="gray")
for i in 1:length(ntaus)
    plt.loglog(ntaus[i], rel_diffs[i], "s", alpha=0.75)
end
plt.xlabel(raw"$N_{\tau}$")
plt.ylabel("Relative Error in ρ")
plt.axis("image")
plt.grid(true)
plt.xlim([1e0, 1e5])

plt.savefig("figure_dimer_convergence.pdf")
plt.show()
