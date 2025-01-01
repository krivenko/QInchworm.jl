# QInchworm.jl
#
# Copyright (C) 2021-2025 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Author: Igor Krivenko

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf

using QInchworm; cfg = QInchworm.configuration

using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.configuration: Configuration, Node, InchNode, NodePair

using QInchworm.qmc_integrate: qmc_time_ordered_integral,
                               qmc_time_ordered_integral_n_samples,
                               qmc_time_ordered_integral_sort,
                               qmc_time_ordered_integral_root

using Sobol: SobolSeq
using LinearAlgebra: norm
using HDF5

import PyPlot; const plt = PyPlot; plt.pygui(false); plt.svg(true);

include("DelaunaySimplex.jl");

β = 10.;         # Inverse temperature
U = 1.0;         # Coulomb interaction
μ = U/2 + 0.3;   # Chemical potential
B = 0.1;         # Magnetic field
#ϵ = [-0.1, +0.1] # Bath energy levels
ϵ = [-1.0, +1.0] # Bath energy levels
V = 1.0;         # Hopping

H = -μ * (op.n(0, "up") + op.n(0, "dn")) + U * op.n(0, "up") * op.n(0, "dn") + B * (op.n(0, "up") - op.n(0, "dn"));
soi = ked.Hilbert.SetOfIndices([[0, "up"], [0, "dn"]]);
ed = ked.EDCore(H, soi);
ρ = ked.density_matrix(ed, β);

function make_second_order_diag(; nt, nτ, tmax, τ_w_pos, verbose = false)
    # 3-branch time contour and a grid on it
    contour = kd.twist(kd.FullContour(tmax=tmax, β=β))
    grid = kd.FullTimeGrid(contour, nt, nτ)

    # Imaginary time grid
    τ_grid = grid[kd.imaginary_branch]
    τ_0, τ_β = τ_grid[1], τ_grid[end]

    # Hybridization function
    dos = kd.DeltaDOS(ϵ, V^2 * ones(length(ϵ)))
    Δ = kd.FullTimeGF(dos, grid);

    if verbose
        τ0 = τ_grid[1]
        vals = [Δ(τ.bpoint, τ0.bpoint) for τ in τ_grid]
        plt.figure()
        plt.title("Δ(τ)")
        plt.plot([-imag(τ.bpoint.val) for τ in τ_grid], imag.(vals), "-");
        plt.xlim(0, β)
    end

    # Pseudo Particle Strong Coupling Expansion
    ip_up = InteractionPair(op.c_dag(0, "up"), op.c(0, "up"), Δ);
    ip_dn = InteractionPair(op.c_dag(0, "dn"), op.c(0, "dn"), Δ);
    ppsc_exp = Expansion(ed, grid, [ip_up, ip_dn]);

    if verbose
        plt.figure()
        plt.title(raw"$\hat P(τ)$")
        for (idx, P_s) in enumerate(ppsc_exp.P0)
            vals = [P_s(τ.bpoint, τ0.bpoint) for τ in τ_grid]
            τ_vals = [-imag(τ.bpoint.val) for τ in τ_grid]
            vals = vcat(vals...)
            plt.plot(τ_vals, imag.(vals), "-", label="\$G^{0}_$idx\$")

            vals = [ppsc_exp.P[idx](τ.bpoint, τ0.bpoint) for τ in τ_grid]
            vals = vcat(vals...)
            plt.plot(τ_vals, imag.(vals), "-", label="\$G_$idx\$")
            plt.xlim(0, β)
        end
        plt.legend(loc="best");
    end

    # 2nd order inchworm diagram on the imaginary branch
    τ_i = τ_0
    τ_f = τ_β

    τ_w_idx = 1 + Int(τ_w_pos * (nτ - 1))
    τ_w = τ_grid[τ_w_idx]

    n_i = Node(τ_i.bpoint)
    n_w = InchNode(τ_w.bpoint)
    n_f = Node(τ_f.bpoint)

    nodes = [n_f, n_w, n_i]
    conf_0 = Configuration(nodes, NodePair[]);
    init_0 = zero(cfg.eval(ppsc_exp, conf_0));

    diagram = τ -> begin
        # Loop over spin indices of hybridization lines
        sum = deepcopy(init_0)
        for index1 = 1:2, index2 = 1:2
            # 4 arrangements of creation and annihilation operators
            conf = Configuration(nodes, [NodePair(n_f.time, τ[2], index1), NodePair(τ[1], τ[3], index2)])
            sum += cfg.eval(ppsc_exp, conf)
            conf = Configuration(nodes, [NodePair(τ[2], n_f.time, index1), NodePair(τ[1], τ[3], index2)])
            sum += cfg.eval(ppsc_exp, conf)
            conf = Configuration(nodes, [NodePair(n_f.time, τ[2], index1), NodePair(τ[3], τ[1], index2)])
            sum += cfg.eval(ppsc_exp, conf)
            conf = Configuration(nodes, [NodePair(τ[2], n_f.time, index1), NodePair(τ[3], τ[1], index2)])
            sum += cfg.eval(ppsc_exp, conf)
        end
        return sum
    end
    (contour, τ_grid, diagram)
end;

function make_ref_value(diagram, contour, τ_grid, τ_w_pos, order_min, order_max; verbose = false)
    b = contour[kd.imaginary_branch]
    τ_vals = [-imag(τ.bpoint.val) for τ in τ_grid]
    τ_w_idx = 1 + Int(τ_w_pos * (length(τ_vals) - 1))

    points, simplices = DelaunaySimplex.triangulate(τ_vals[1:τ_w_idx], 3);

    func_sector(z, y, x, sector) = -imag(diagram([b(z/β), b(y/β), b(x/β)])[sector][2][1, 1])
    scalar_integrand(x, sector) = func_sector(x[1], x[2], x[3], sector)

    orders = collect(order_min:1:order_max)
    delaunay_res = Array{Float64, 2}(undef, 0, 4)

    for order = orders
        val = [
            DelaunaySimplex.integrate_t3(points, simplices, order) do x
                [scalar_integrand(x[:, i], sector) for i in eachindex(x, 2)]
            end
            for sector = 1:4
        ]
        if verbose
           println("val($order)    = $val")
            if length(delaunay_res) > 0
                diff = val - delaunay_res[end, :]
                println("diff      = $diff")
            end
        end
        delaunay_res = cat(delaunay_res, val'; dims=1)
    end

    if verbose
        delaunay_errors = abs.(delaunay_res[1:end-1, :] .- delaunay_res[end, :]')
        plt.semilogy(orders[2:end], delaunay_errors)
        plt.xlabel("integrator order")
        plt.ylabel("pairwise difference")
        plt.savefig("figure_delaunay_integration_convergence.pdf")
    end

    delaunay_res[end, :]
end;

function h5_write_results(group_name, ref, nτ, N_list, results, div)
    h5open("2nd_order_inchworm_hpc.h5", "cw") do file
        try
            delete_object(file[group_name])
        catch KeyError end
        g = create_group(file, group_name)
        write(g, "ref", ref)
        write(g, "ntau", nτ)
        write(g, "N_list", N_list)
        write(g, "results", results)
        write(g, "div", div)
    end
end

function scan_N_range_root(diagram, contour, τ_grid, τ_w_pos, N_list::Vector{Int}; verbose = false)
    τ_i = τ_grid[1]
    τ_w_idx = 1 + Int(τ_w_pos * (length(τ_grid) - 1))
    τ_w = τ_grid[τ_w_idx]

    N_steps = [N_list[1]]
    append!(N_steps, diff(N_list))
    N_skip_list = [0]
    append!(N_skip_list, N_list[1:end-1])

    chunks = zeros(Complex{Float64}, 4, length(N_steps))

    seq = SobolSeq(3)
    i_N_steps = collect(Iterators.enumerate(N_steps))
    Threads.@threads for (i, N_step) in i_N_steps
        N_skip = N_skip_list[i]
        seq = skip(SobolSeq(3), N_skip, exact = true)
        chunk = qmc_time_ordered_integral_root(diagram, 3, contour,
                                               τ_i.bpoint, τ_w.bpoint,
                                               init = deepcopy(diagram.init_0),
                                               seq = seq, N = N_step)
        chunk *= N_step
        chunks[:, i] = [chunk[s][2][1] for s = 1:4]

        if verbose
            tid = Threads.threadid()
            println("[$tid] Chunk $i: Sobol sequence points [$(N_skip + 1); $(N_skip + 1 + N_step)[: contribution = $(chunks[:, i])")
            flush(stdout)
        end
    end

    results = cumsum(chunks, dims=2)
    for i = 1:length(N_steps)
        results[:, i] /= N_list[i]
    end
    results
end;

nτ_list = [11]
#nτ_list = [21]
τ_w_pos = 0.8

N_list = round.(Int, exp10.(range(2, 6, length=500)))
#N_list = round.(Int, exp10.(range(2, 8, length=1000)))

for nτ in nτ_list
    contour, τ_grid, diagram = make_second_order_diag(nt = 2, nτ = nτ, tmax = 1.0, τ_w_pos = 0.8)
    @time ref = make_ref_value(diagram, contour, τ_grid, τ_w_pos, 9, 9)
    println("ref = $ref")

    @time results = scan_N_range_root(diagram, contour, τ_grid, τ_w_pos, N_list, verbose = true)
    div = results - (ref .* ones(ComplexF64, length(N_list))')

    h5_write_results("scan_N_range_root/ntau$(nτ)", ref, nτ, N_list, results, div)
end
