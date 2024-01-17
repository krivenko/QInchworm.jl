# QInchworm.jl
#
# Copyright (C) 2021-2024 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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

filenames = filter(f -> occursin("data_", f), readdir(".", join=true))

for filename in filenames
    @show filename
    fid = h5.h5open(filename, "r")
    g = fid["data"]
    d = read_group(g)
    h5.close(fid)

    orders = first(d["orders"]):last(d["orders"])
    N_samples = d["N_samples"]
    N_seqs = d["N_seqs"]

    β = d["beta"]
    τ = d["tau"]

    plt.figure(figsize=(3.25*2, 8))

    # Plot PPGF
    plt.subplot(2, 1, 1)
    for n = 1:2
        P = d["P"][string(n)]
        P_std = d["P_std"][string(n)]
        plt.errorbar(τ, -imag(P), yerr=real(P_std), lw=0.5, label="state $(n)")
    end
    plt.title("orders = $(orders), N_samples = $(N_samples), N_seqs = $(N_seqs)")
    plt.xlim((0, β))
    plt.xlabel(raw"$\tau$")
    plt.ylabel(raw"$-P(\tau)$")
    plt.legend()
    plt.tight_layout()

    # Plot GF
    plt.subplot(2, 1, 2);
    plt.errorbar(τ, -imag(d["gf"]), yerr=real(d["gf_std"]),
                lw=0.5, color="red", label="RqMC")
    plt.plot(τ, imag(d["gf_ref"]), lw=0.5, color="blue", label="ref")
    plt.title("orders = $(orders), N_samples = $(N_samples), N_seqs = $(N_seqs)")
    plt.xlim((0, β))
    plt.xlabel(raw"$\tau$")
    plt.ylabel(raw"$-G(\tau)$")
    plt.legend()
    plt.tight_layout()

    plt.savefig("figure_N_samples_$(N_samples)_N_seqs_$(N_seqs)_orders_$(orders).pdf")
end
