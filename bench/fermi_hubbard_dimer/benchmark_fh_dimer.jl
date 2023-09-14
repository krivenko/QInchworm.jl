using MPI

using MD5
using HDF5; h5 = HDF5

using Printf

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf
using QInchworm.diagrammatics: get_topologies_at_order
using QInchworm.expansion: Expansion, InteractionPair, get_diagrams_at_order

using QInchworm.inchworm: inchworm!
using QInchworm.mpi: ismaster


function run_hubbard_dimer(ntau, orders, orders_bare, N_samples)

    β = 1.0
    U = 4.0
    ϵ_1, ϵ_2 = 0.0 - 0.5*U, 2.0
    V_1 = 0.5
    V_2 = 0.5

    # -- ED solution

    H_imp = U * op.n(1) * op.n(2) + ϵ_1 * (op.n(1) + op.n(2))

    H_dimer = H_imp + ϵ_2 * (op.n(3) + op.n(4)) +
        V_1 * ( op.c_dag(1) * op.c(3) + op.c_dag(3) * op.c(1) ) +
        V_2 * ( op.c_dag(2) * op.c(4) + op.c_dag(4) * op.c(2) )

    soi_dimer = ked.Hilbert.SetOfIndices([[1], [2], [3], [4]])
    ed_dimer = ked.EDCore(H_dimer, soi_dimer)

    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    ρ_ref = Array{ComplexF64}( reduced_density_matrix(ed_dimer, soi, β) )

    # -- Hybridization propagator

    Δ_1 = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V_1^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ_2, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ_2),
        grid, 1, kd.fermionic, true)

    Δ_2 = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V_2^2 *
            (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ_2, contour.β)) *
            exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ_2),
        grid, 1, kd.fermionic, true)

    function reverse(g::kd.ImaginaryTimeGF)
        g_rev = deepcopy(g)
        τ_0, τ_β = first(g.grid), last(g.grid)
        for τ in g.grid
            g_rev[τ, τ_0] = g[τ_β, τ]
        end
        return g_rev
    end

    # -- Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ_1)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ_1))
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ_2)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ_2))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    ρ_0 = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P0), ed), ed)

    inchworm!(expansion,
              grid,
              orders,
              orders_bare,
              N_samples)

    ppgf.normalize!(expansion.P, β)
    ρ_wrm = full_hs_matrix(tofockbasis(ppgf.density_matrix(expansion.P), ed), ed)
    diff = maximum(abs.(ρ_ref - ρ_wrm))

    if ismaster()
        @printf "ρ_0   = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2]) real(ρ_0[3, 3]) real(ρ_0[4, 4])
        @printf "ρ_ref = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2]) real(ρ_ref[3, 3]) real(ρ_ref[4, 4])
        @printf "ρ_wrm = %16.16f %16.16f %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2]) real(ρ_wrm[3, 3]) real(ρ_wrm[4, 4])
        @show diff
    end

    return diff
end

function run_ntau_calc(ntau::Integer, orders, N_sampless)

    comm_root = 0
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    orders_bare = orders

    # -- Do calculation here

    diff_0 = run_hubbard_dimer(ntau, orders, orders_bare, 0)

    diffs = [ run_hubbard_dimer(ntau, orders, orders_bare, N_samples)
              for N_samples in N_sampless ]

    if comm_rank == comm_root
        @show diffs

        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        max_order = maximum(orders)
        filename = "data_FH_dimer_ntau_$(ntau)_maxorder_$(max_order)_md5_$(id).h5"

        @show filename
        fid = h5.h5open(filename, "w")

        g = h5.create_group(fid, "data")

        h5.attributes(g)["ntau"] = ntau
        h5.attributes(g)["diff_0"] = diff_0

        g["orders"] = collect(orders)
        g["orders_bare"] = collect(orders_bare)
        g["N_sampless"] = N_sampless

        g["diffs"] = diffs

        h5.close(fid)

    end
end


MPI.Init()

#ntaus = 2 .^ (4:8)
#N_sampless = 8 * 2 .^ (1:10)
#orderss = [0:1, 0:2, 0:3]

#ntaus = 2 .^ (4:6)
#ntaus = 2 .^ (4:8)
#ntaus = 2 .^ (7:8)
ntaus = 2 .^ (8:8)
#orderss = [0:1]
orderss = [0:2, 0:3]
#N_sampless = 2 .^ (3:14)
N_sampless = 2 .^ (15:17)

if ismaster()
    @show ntaus
    @show N_sampless
    @show orderss
end

# exit()

for ntau in ntaus
    for orders in orderss
        run_ntau_calc(ntau, orders, N_sampless)
    end
end
