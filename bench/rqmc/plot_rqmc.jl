using Glob: glob
using HDF5; h5 = HDF5;

using PyPlot; plt = PyPlot;

function read_group(group)
    return merge(
        Dict(key => h5.read(group, key) for key in keys(group)),
        Dict(key => h5.read_attribute(group, key) for key in keys(h5.attributes(group)))
    )
end

filenames = filter(f -> occursin("data_", f), readdir(".", join=true))
@show filenames

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
