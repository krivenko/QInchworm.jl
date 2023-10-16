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
