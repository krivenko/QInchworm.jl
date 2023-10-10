using HDF5; h5 = HDF5

using PyPlot; plt = PyPlot

filenames = filter(f -> endswith(f, ".h5"), readdir(".", join=true))

ds = []
for filename in filenames
    println("--> Loading: $filename")
    d = Dict()
    h5.h5open(filename, "r") do fid
        grp = fid["data"]
        for key in keys(h5.attributes(grp))
            d[key] = h5.read_attribute(grp, key)
        end
        for key in keys(grp)
            d[key] = h5.read(grp, key)
        end
    end
    push!(ds, d)
end

plt.figure()
subp = [3, 1, 1]

plt.subplot(subp...); subp[end] += 1
for d in ds
    maxorder = d["maxorder"][1]
    plt.plot(d["tau"], d["P1"], "r", label="P1 order $maxorder")
    plt.plot(d["tau"], d["P2"], "g", label="P2 order $maxorder")
    plt.plot(d["tau"], d["P1_exact"], "c--")
    plt.plot(d["tau"], d["P2_exact"], "m--")
end

plt.subplot(subp...); subp[end] += 1
for d in ds
    maxorder = d["maxorder"][1]
    plt.plot(d["tau"], d["P1"] - d["P1_exact"], label="P1 order $maxorder")
    plt.plot(d["tau"], d["P2"] - d["P2_exact"], label="P2 order $maxorder")
end

plt.subplot(subp...); subp[end] += 1
for d in ds
    maxorder = d["maxorder"][1]
    plt.plot(d["tau"], abs.(d["P1"] - d["P1_exact"]), label="P1 order $maxorder")
    plt.plot(d["tau"], abs.(d["P2"] - d["P2_exact"]), label="P2 order $maxorder")
end

plt.semilogy([], [])
plt.legend(loc="best")
plt.tight_layout()
plt.show()
