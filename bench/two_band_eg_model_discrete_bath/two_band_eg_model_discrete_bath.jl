"""

Author: Hugo U. R. Strand (2023)

"""

using MD5
using HDF5; h5 = HDF5

using Test
using LinearInterpolations: Interpolate

using MPI; MPI.Init()

#import PyPlot; plt = PyPlot

using LinearAlgebra: diag
using QuadGK: quadgk

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: normalize!, density_matrix, atomic_ppgf!
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p
using QInchworm.mpi: ismaster

function make_hamiltonian(n_orb, mu, U, J)
  soi = ked.Hilbert.SetOfIndices([[s,o] for s in ("up","dn") for o = 1:n_orb])

  H = op.OperatorExpr{Float64}()

  for o=1:n_orb; H += -mu * (op.n("up", o) + op.n("dn", o)) end
  for o=1:n_orb; H += U * op.n("up", o) * op.n("dn", o) end

  for o1=1:n_orb, o2=1:n_orb
    o1 == o2 && continue
    H += (U - 2 * J) * op.n("up", o1) * op.n("dn", o2)
  end
  for o1=1:n_orb, o2=1:n_orb
    o2 >= o1 && continue
    H += (U - 3 * J) * op.n("up", o1) * op.n("up", o2)
    H += (U - 3 * J) * op.n("dn", o1) * op.n("dn", o2)
  end
  for o1=1:n_orb, o2=1:n_orb
    o1 == o2 && continue
    H += -J * op.c_dag("up", o1) * op.c_dag("dn", o1) * op.c("up", o2) * op.c("dn", o2)
    H += -J * op.c_dag("up", o1) * op.c_dag("dn", o2) * op.c("up", o2) * op.c("dn", o1)
  end

  (soi, H)
end

function kernel(t, w)
    if w > 0
        return exp(-t * w) / (1 + exp(-w))
    else
        return exp((1 - t)*w) / (1 + exp(w))
    end
end

function semi_circular_g_tau(times, t, h, β)

    g_out = zero(times)

    function kernel(t, w)
        if w > 0
            return exp(-t * w) / (1 + exp(-w))
        else
            return exp((1 - t)*w) / (1 + exp(w))
        end
    end

    for (i, τ) in enumerate(times)
        I = x -> -2 / pi / t^2 * kernel(τ/β, β*x) * sqrt(x + t - h) * sqrt(t + h - x)
        g, err = quadgk(I, -t+h, t+h; rtol=1e-12)
        g_out[i] = g
    end

    return g_out
end

function run_bethe(nτ, orders, orders_bare, orders_gf, N_samples, n_pts_after_max; discrete_bath=true)

    n_orb = 2
    β = 8.0

    t_bethe = 2.0
    μ_bethe = 0.0

    e_k = 2.3

    U = 2.0
    J = 0.2
    μ = (3*U - 5*J)/2 - 1.5

    # -- ED solution

    soi, H_imp = make_hamiltonian(n_orb, μ, U, J)
    #println(H_imp)
    #println(soi)

    # -- Need to break symmetry of ED! FIXME

    esgs = [
        op.c_dag("up", 1) * op.c("up", 2) + op.c_dag("up", 2) * op.c("up", 1),
        op.c_dag("dn", 1) * op.c("dn", 2) + op.c_dag("dn", 2) * op.c("dn", 1),
    ]

    #K = 1e-9

    #H_imp += K * op.c_dag("up", 1) * op.c("up", 2)
    #H_imp += K * op.c_dag("up", 2) * op.c("up", 1)

    #H_imp += K * op.c_dag("dn", 1) * op.c("dn", 2)
    #H_imp += K * op.c_dag("dn", 2) * op.c("dn", 1)

    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, nτ);
    ed = ked.EDCore(H_imp, soi, symmetry_breakers=esgs)

    # -- Hybridization propagator

    #tau = [ real(im * τ.bpoint.val) for τ in grid ]
    #delta_bethe = V^2 * semi_circular_g_tau(tau, t_bethe, μ_bethe, β)

    if discrete_bath

        if ismaster(); println("--> Discrete Bath"); end

        Δ = kd.ImaginaryTimeGF(
            (t1, t2) -> -1.0im * (
                    kernel(-imag(t1.bpoint.val - t2.bpoint.val) / β, +e_k * β) +
                    kernel(-imag(t1.bpoint.val - t2.bpoint.val) / β, -e_k * β) ),
            grid, 1, kd.fermionic, true)
    else

        if ismaster(); println("--> Bethe Bath"); end

        Δ = kd.ImaginaryTimeGF(
            (t1, t2) -> 1.0im *
                semi_circular_g_tau(
                    [-imag(t1.bpoint.val - t2.bpoint.val)],
                    t_bethe, μ_bethe, β)[1],
            grid, 1, kd.fermionic, true)
    end


    function reverse(g::kd.ImaginaryTimeGF)
        g_rev = deepcopy(g)
        τ_0, τ_β = first(g.grid), last(g.grid)
        for τ in g.grid
            g_rev[τ, τ_0] = g[τ_β, τ]
        end
        return g_rev
    end

    # -- Pseudo Particle Strong Coupling Expansion

    ips = Array{InteractionPair{kd.ImaginaryTimeGF{ComplexF64, true}}, 1}()

    for s in ["up", "dn"], o in [1, 2]
        push!(ips, InteractionPair(op.c_dag(s, o), op.c(s, o), Δ))
        push!(ips, InteractionPair(op.c(s, o), op.c_dag(s, o), reverse(Δ)))
    end

    for s in ["up", "dn"]
        push!(ips, InteractionPair(op.c_dag(s, 1), op.c(s, 2), Δ))
        push!(ips, InteractionPair(op.c(s, 2), op.c_dag(s, 1), reverse(Δ)))

        push!(ips, InteractionPair(op.c_dag(s, 2), op.c(s, 1), Δ))
        push!(ips, InteractionPair(op.c(s, 1), op.c_dag(s, 2), reverse(Δ)))
    end

    #println(ips)

    expansion = Expansion(ed, grid, ips)
    #atomic_ppgf!(expansion.P0, ed, Δλ=1.0)
    atomic_ppgf!(expansion.P0, ed, Δλ=2.0)

    inchworm!(expansion, grid, orders, orders_bare, N_samples;
              n_pts_after_max=n_pts_after_max)

    P_raw = deepcopy(expansion.P)
    normalize!(expansion.P, β)

    add_corr_operators!(expansion, (op.c("up", 1), op.c_dag("up", 1)))
    add_corr_operators!(expansion, (op.c("dn", 1), op.c_dag("dn", 1)))

    add_corr_operators!(expansion, (op.c("up", 2), op.c_dag("up", 2)))
    add_corr_operators!(expansion, (op.c("dn", 2), op.c_dag("dn", 2)))

    add_corr_operators!(expansion, (op.c("up", 1), op.c_dag("up", 2)))
    add_corr_operators!(expansion, (op.c("dn", 1), op.c_dag("dn", 2)))

    add_corr_operators!(expansion, (op.c("up", 2), op.c_dag("up", 1)))
    add_corr_operators!(expansion, (op.c("dn", 2), op.c_dag("dn", 1)))

    g = correlator_2p(expansion, grid, orders_gf, N_samples)

    # ==

    if ismaster()
        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, vcat(g[1].mat.data...))))
        filename = "data_order_$(orders)_ntau_$(nτ)_N_samples_$(N_samples)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")
        grp = h5.create_group(fid, "data")

        h5.attributes(grp)["beta"] = β
        h5.attributes(grp)["ntau"] = nτ
        h5.attributes(grp)["n_pts_after_max"] = n_pts_after_max
        h5.attributes(grp)["N_samples"] = N_samples

        grp["orders"] = collect(orders)
        grp["orders_bare"] = collect(orders_bare)
        grp["orders_gf"] = collect(orders_gf)

        grp["tau"] = collect(kd.imagtimes(g[1].grid))

        grp["gf_up_11"] = g[1].mat.data[1, 1, :]
        grp["gf_dn_11"] = g[2].mat.data[1, 1, :]

        grp["gf_up_22"] = g[3].mat.data[1, 1, :]
        grp["gf_dn_22"] = g[4].mat.data[1, 1, :]

        grp["gf_up_12"] = g[5].mat.data[1, 1, :]
        grp["gf_dn_12"] = g[6].mat.data[1, 1, :]

        grp["gf_up_21"] = g[7].mat.data[1, 1, :]
        grp["gf_dn_21"] = g[8].mat.data[1, 1, :]

        for (s, p) in enumerate(expansion.P)
            grp["P_$(s)"] = p.mat.data
            grp["P0_$(s)"] = expansion.P0[s].mat.data
            grp["Praw_$(s)"] = P_raw[s].mat.data
        end

        grp["gf_ref"] = -Δ.mat.data[1, 1, :]

        h5.close(fid)
    end

    #if ismaster()
    if false

        τ = kd.imagtimes(g[1].grid)
        τ_ref = collect(LinRange(0, β, 128))

        plt.figure(figsize=(3.25*2, 8))
        subp = [3, 2, 1]

        for s in 1:length(expansion.P)
            plt.subplot(subp...); subp[end] += 1;

            x = collect(τ)
            y = collect(imag(expansion.P[s].mat.data[1, 1, :]))
            P_int = Interpolate(x, y)
            P = P_int.(τ_ref)

            plt.plot(τ_ref, -P, label="P$(s)")
            plt.semilogy([], [])
            plt.ylabel(raw"$P_\Gamma(\tau)$")
            plt.xlabel(raw"$\tau$")
            plt.legend(loc="best")

            plt.subplot(subp...); subp[end] += 1;
            for (o, P) in enumerate(expansion.P_orders)
                p = imag(P[s].mat.data[1, 1, :])
                plt.semilogy(τ, -p, label="order $(o-1) ref", alpha=0.25)
            end
            plt.ylim(bottom=1e-9)
            plt.ylabel(raw"$P_\Gamma(\tau)$")
            plt.xlabel(raw"$\tau$")
            plt.legend(loc="best")
        end

        x = collect(τ)
        y = collect(imag(g[1].mat.data[1, 1, :]))
        g_int = Interpolate(x, y)
        gr = g_int.(τ_ref)

        plt.subplot(subp...); subp[end] += 1;
        plt.title("nτ = $(length(τ)), N_samples = $N_samples")
        plt.plot(τ, imag(g[1].mat.data[1, 1, :]), "--", label="InchW")
        plt.plot(τ, -imag(Δ.mat.data[1, 1, :])/V^2, "--", label="Bethe")
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$G_{11}(τ)$")
        plt.legend(loc="best")
        plt.ylim(bottom=0)

        plt.tight_layout()
        plt.savefig("figure_ntau_$(nτ)_N_samples_$(N_samples)_orders_$(orders).pdf")
        #plt.show()
    end
end



@assert length(ARGS) == 4

order = parse(Int, ARGS[1])
nτ = parse(Int, ARGS[2])
N_samples = parse(Int, ARGS[3])
n_pts_after_max = parse(Int, ARGS[4])

if n_pts_after_max == 0
    n_pts_after_max = typemax(Int64)
end

order_gf = order - 1

if ismaster()
    println("order $(order) nτ $(nτ) N_samples $(N_samples) n_pts_after_max $(n_pts_after_max)")
end

#exit()

orders = 0:order
orders_gf = 0:order_gf

run_bethe(nτ, orders, orders, orders_gf, N_samples, n_pts_after_max)
