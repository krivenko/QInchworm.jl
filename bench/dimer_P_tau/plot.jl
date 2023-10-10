using HDF5; h5 = HDF5
using DataFrames

using PyPlot; plt = PyPlot

filenames = filter(f -> endswith(f, ".h5"), readdir(".", join=true))

df = DataFrame()

for filename in filenames
    println("--> Loading: $filename")
    d = Dict()
    h5.h5open(filename, "r") do fid
        grp = fid["data"]
        for key in keys(h5.attributes(grp))
            d[key] = h5.read_attribute(grp, key)
        end
    end
    ndf = DataFrame(d)
    global df = vcat(df, ndf)
end

@show df

maxorders = sort(unique(df[:, :maxorder]))

styles = Dict(
    1. => "o",
    3. => "s",
    4. => "D",
    5. => "+",
    6. => "x"
)

plt.figure(figsize=(10, 9))

colors = Dict()
for nτ in sort(unique(df[:, :ntau]))
    colors[nτ] = plt.plot([], [])[1].get_color()
end

for maxorder in maxorders
    local nτs = sort(unique(df[(df.maxorder .== maxorder), :ntau]))
    style = styles[maxorder]
    for nτ in nτs
        color = colors[nτ]
        ns = df[(df.maxorder .== maxorder) .& (df.ntau .== nτ), :N_samples]
        diff = df[(df.maxorder .== maxorder) .& (df.ntau .== nτ), :diff]
        sidx = sortperm(ns)
        plt.plot(ns[sidx], diff[sidx],
                 "-" * style, alpha=0.5, color=color,
                 label="order $maxorder \$n\\tau\$ $nτ")
    end
end

plt.loglog([], [])
plt.legend(loc="lower left", ncol=1)
plt.grid(true)
plt.xlabel("N samples")
plt.ylabel(raw"$\max|\Delta P(\tau)|$")

plt.tight_layout()
plt.savefig("figure_dimer_P_tau_cf.pdf")
plt.show()
