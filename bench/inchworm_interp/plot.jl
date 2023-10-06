using PyPlot; plt = PyPlot
using HDF5; h5 = HDF5

using PyCall
PdfPages = pyimport("matplotlib.backends.backend_pdf").PdfPages

fid = h5.h5open("data.h5", "r")
data_g = h5.open_group(fid, "data")

μ_bethe = h5.attributes(data_g)["mu_bethe"]
N_samples = h5.attributes(data_g)["N_samples"]

pdf = PdfPages("figure_inchworm_interp.pdf")

for g in data_g
    nτ_list = read(h5.attributes(g)["ntau_list"])
    orders = read(g["orders"])
    diffs_non_interp = read(g["diffs_non_interp"])
    diffs_interp = read(g["diffs_interp"])

    plt.figure()
    cmap = plt.get_cmap("tab10")
    for (i, name) in enumerate(["NCA", "OCA", "2CA", "exact"])
        plt.loglog(nτ_list, diffs_non_interp[:, i], base=2,
            lw = 0.5, ls="--", color=cmap(i),
            label = "diff(\$\\rho_{$name}\$), linear interp.")
        plt.loglog(nτ_list, diffs_interp[:, i], base=2,
            lw = 0.5, color=cmap(i),
            label = "diff(\$\\rho_{$name}\$), quadratic interp.")
    end
    plt.xscale("log", base = 2)
    plt.yscale("log", base = 10)
    plt.xlabel(raw"$N_\tau$")
    plt.title("Expansion orders " * join(orders, ", "))
    plt.legend(prop=Dict("size" => 8))
    pdf.savefig()
end

pdf.close()
