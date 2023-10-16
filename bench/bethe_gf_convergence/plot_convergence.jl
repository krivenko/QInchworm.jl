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

using HDF5; h5 = HDF5
using DataFrames

using PyPlot; plt = PyPlot

function read_and_sort(filenames)

    ps = []
    df = DataFrame()

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

        d["error"] = maximum(abs.(d["gf"] - d["gf_ref"]))
        d["order"] = maximum(d["orders"])

        push!(ps, copy(d))

        pop!(d, "gf")
        pop!(d, "gf_ref")
        pop!(d, "tau")

        pop!(d, "orders")
        pop!(d, "orders_bare")
        pop!(d, "orders_gf")

        ndf = DataFrame(d)
        df = vcat(df, ndf)
    end

    order = [maximum(p["orders"]) for p in ps]
    sidx = sortperm(order)
    ps = ps[sidx]

    return ps, df
end

paths = [
#    "./n_pts_after_max_1/data",
#    "./n_pts_after_max_NoLimit/data",
    "./"
]

filenames = filter(f -> endswith(f, ".h5"), readdir(paths[1], join=true))
ps, df = read_and_sort(filenames)
@show df

plt.figure(figsize=(3.25 * 2, 5))

nτs = sort(unique(df[:, "ntau"]))
orders = sort(unique(df[:, "order"]))

styles = ["o", "s", "d", ">", "<", "^", "x", "+"]

colors = []
for nτ in nτs
    c = plt.plot([], [], label="\$n\\tau\$ $nτ")[1].get_color()
    push!(colors, c)
end

for (style, order) in zip(styles, orders)
    plt.plot([], [], style, color="gray", label="order $order")
end

for (style, order) in zip(styles, orders)
    for (color, nτ) in zip(colors, nτs)
        dfc = df[(df.ntau .== nτ) .& (df.order .== order), :]
        sort!(dfc, :N_samples)

        plt.plot(
            dfc[:, "N_samples"], dfc[:, "error"],
            "-" * style, alpha=0.5, color=color
        )
    end
end

plt.legend()
plt.loglog([], [])
plt.grid(true)

plt.show()

plt.figure(figsize=(3.25, 5))
subp = [2, 1, 1]

plt.subplot(subp...); subp[end] += 1

for p in ps
    order = maximum(p["orders"])
    plt.plot(p["tau"], imag(p["gf"]), label="order $order", alpha=0.75)
end

p = ps[1]
plt.plot(p["tau"], imag(p["gf_ref"]), "--k", label="Analytic")
plt.ylim(bottom=0)
plt.legend(loc="best")
plt.xlabel(raw"$\tau$")
plt.ylabel(raw"$G(\tau)$")
plt.grid(true)

plt.subplot(subp...); subp[end] += 1

for path in paths
    local filenames = filter(f -> endswith(f, ".h5"), readdir(path, join=true))
    local ps, df = read_and_sort(filenames)

    order = [maximum(p["orders"]) for p in ps]
    err = [maximum(abs.(p["gf"] - p["gf_ref"])) for p in ps]
    if ps[1]["n_pts_after_max"] == 1
        label = "n_pts_after_max = 1"
    else
        label = "n_pts_after_max = unrestricted"
    end

    plt.plot(order, err, "-o", alpha=0.75, label=label)
end

plt.xlabel("order")
plt.ylabel("GF error")
plt.legend(loc="best")
plt.semilogy([], [])
plt.grid(true)

plt.tight_layout()
plt.savefig("figure_bethe_gf_convergence.pdf")

plt.show()
