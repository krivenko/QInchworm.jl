
import PyPlot as plt
import HDF5; h5 = HDF5


function read_group(group)
    return merge(
        Dict( key => h5.read(group, key) for key in keys(group)),
        Dict( key => h5.read_attribute(group, key) for key in keys(h5.attributes(group)) ) )  
end


filenames = filter( f -> occursin(".h5", f), readdir(".", join=true) )
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

# Plot for all ntau

fig = plt.figure(figsize=(3.24 * 2, 3))
gs = fig.add_gridspec(
    nrows=1, ncols=2,
    width_ratios=(1.5, 1),
    left=0.10, right=0.99,
    top=0.95, bottom=0.20,
    wspace=0.3)

#subp = [1, 2, 1]

#plt.subplot(subp...); subp[end] += 1
plt.subplot(gs[1, 1])

for key in sort(collect(keys(merged_data)))
    d = merged_data[key]
    ntau = d["ntau"]
    #N = d["N_chunkss"] .* d["ntau"] .* d["N_per_chunk"]
    N = d["N_chunkss"] .* d["N_per_chunk"]
    rel_diffs = d["diffs"] ./ d["diff_0"]
    plt.loglog(N, rel_diffs, "-o", label=raw"$n_{\tau}$" * " = $ntau", alpha=0.5)
end
    
plt.legend(fontsize=7, loc="best")
plt.xlabel(raw"$N_{QQMC, tot} / N_{\tau}$")
plt.ylabel("Relative Error in ρ")
plt.axis("image")
plt.grid(true)

#plt.subplot(subp...); subp[end] += 1
plt.subplot(gs[1, 2])

ntaus = sort(collect(keys(merged_data)))
rel_diffs = [ d["diffs"][end] ./ d["diff_0"] for d in [ merged_data[ntau] for ntau in ntaus ] ]

for i in 1:length(ntaus)
    plt.loglog(ntaus[i], rel_diffs[i], "s", alpha=0.5)
end

plt.loglog(ntaus, rel_diffs, "-", color="gray", zorder=-100)
plt.xlabel(raw"$N_{\tau}$")
plt.ylabel("Relative Error in ρ")
plt.axis("image")
plt.grid(true)

plt.savefig("figure_dimer_convergence.pdf")
plt.show()
