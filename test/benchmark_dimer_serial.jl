using MPI
using Test
using Printf
import PyPlot as plt

import LinearAlgebra; trace = LinearAlgebra.tr

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf
import QInchworm.configuration: Expansion, InteractionPair
import QInchworm.topology_eval: get_topologies_at_order,
                                get_diagrams_at_order
import QInchworm.inchworm: InchwormOrderData,
                           inchworm_step,
                           inchworm_step_bare,
                           inchworm_matsubara!
import QInchworm.KeldyshED_addons: reduced_density_matrix, density_matrix
import QInchworm.spline_gf: SplineInterpolatedGF
using  QInchworm.utility: inch_print


function run_dimer(ntau, orders, orders_bare, N_chunk, max_chunks, qmc_convergence_atol; interpolate_gfs=false)

    if inch_print(); @show interpolate_gfs; end
    
    β = 1.0
    ϵ_1, ϵ_2 = 0.0, 2.0
    V = 0.5

    # -- ED solution

    H_dimer = ϵ_1 * op.n(1) + ϵ_2 * op.n(2) + V * ( op.c_dag(1) * op.c(2) + op.c_dag(2) * op.c(1) )
    soi_dimer = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed_dimer = KeldyshED.EDCore(H_dimer, soi_dimer)
    
    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    
    H = ϵ_1 * op.n(1)
    soi = KeldyshED.Hilbert.SetOfIndices([[1]])
    ed = KeldyshED.EDCore(H, soi)

    ρ_ref = Array{ComplexF64}( reduced_density_matrix(ed_dimer, ed, β) )
    
    # -- Hybridization propagator
    
    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> -1.0im * V^2 *
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

    if interpolate_gfs
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), SplineInterpolatedGF(Δ))
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), SplineInterpolatedGF(reverse(Δ)))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=true)
    else
        ip_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
        ip_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
        expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])
    end
    
    ρ_0 = density_matrix(expansion.P0, ed)
    
    inchworm_matsubara!(expansion,
                        grid,
                        orders,
                        orders_bare,
                        N_chunk,
                        max_chunks,
                        qmc_convergence_atol)

    if interpolate_gfs
        P = [ p.GF for p in expansion.P ]
        ppgf.normalize!(P, β) # DEBUG fixme!
        ρ_wrm = density_matrix(P, ed)
    else
        ppgf.normalize!(expansion.P, β) # DEBUG fixme!
        ρ_wrm = density_matrix(expansion.P, ed)
    end

    #ppgf.normalize!(expansion.P, β)
    #ρ_wrm = density_matrix(expansion.P, ed)

    diff = maximum(abs.(ρ_ref - ρ_wrm))

    if inch_print()
        @printf "ρ_0   = %16.16f %16.16f \n" real(ρ_0[1, 1]) real(ρ_0[2, 2])
        @printf "ρ_ref = %16.16f %16.16f \n" real(ρ_ref[1, 1]) real(ρ_ref[2, 2])
        @printf "ρ_wrm = %16.16f %16.16f \n" real(ρ_wrm[1, 1]) real(ρ_wrm[2, 2])
    
        @show diff
    end
    return diff
end


@testset "ntau_plot" begin

    MPI.Init()

    comm_root = 0
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    
    orders = 0:1
    orders_bare = 0:1
    qmc_convergence_atol = 1e-15
    N_per_chunk = 8
    #ntau = 4
    #ntau = 8
    #ntau = 16
    #ntau = 32
    ntau = 64
    #ntau = 128
    #ntau = 256
    #ntau = 512
    #ntau = 1024
    #ntau = 2048
    #ntau = 4096
    #ntau = 4096*2
    #ntau = 4096*4
    #ntau = 4096*8

    #N_chunkss = unique(trunc.(Int, 2 .^ (range(2, 13))))
    N_chunkss = unique(trunc.(Int, 2 .^ (range(0, 13))))
    #N_chunkss = unique(trunc.(Int, 2 .^ (range(4, 4))))
    if inch_print(); @show N_chunkss; end

    #exit()

    #N_chunkss = unique(trunc.(Int, 2 .^ (range(0, 8, 40))))
    #N_chunkss = unique(trunc.(Int, 2 .^ (range(8, 10, 8*2))))
    #N_chunkss = unique(trunc.(Int, 2 .^ (range(0, 10, 8*2 + 8*5))))

    #N_chunkss = unique(trunc.(Int, 2 .^ (range(10, 12, 8*2))))
    #N_chunkss = unique(trunc.(Int, 2 .^ (range(0, 12, 8*(2 + 2 + 5)))))

    #N_chunkss = unique(trunc.(Int, 2 .^ (range(12, 14, 8*2))))

    #N_chunkss = [1024 / 8 / 8]


    # -- Do calculation here
    
    #local_diffs = zeros(Float64, local_size)
    #local_diffs .= 0.1 * comm_rank

    diffs = [ run_dimer(ntau, orders, orders_bare, N_per_chunk, N_chunks,
                              qmc_convergence_atol, interpolate_gfs=false) for N_chunks in N_chunkss ]


    diff_0 = run_dimer(ntau, orders, orders_bare, N_per_chunk, 0, qmc_convergence_atol, interpolate_gfs=false)

    if comm_rank == comm_root
        @show diffs

        import MD5
        import HDF5; h5 = HDF5

        id = MD5.bytes2hex(MD5.md5(reinterpret(UInt8, diffs)))
        filename = "data_ntau_$(ntau)_md5_$(id).h5"
        
        @show filename
        fid = h5.h5open(filename, "w")
        
        g = h5.create_group(fid, "data")
        
        h5.attributes(g)["qmc_convergence_atol"] = qmc_convergence_atol
        h5.attributes(g)["ntau"] = ntau
        h5.attributes(g)["N_per_chunk"] = N_per_chunk
        h5.attributes(g)["diff_0"] = diff_0
        
        g["orders"] = collect(orders)
        g["orders_bare"] = collect(orders_bare)
        g["N_chunkss"] = N_chunkss
        
        g["diffs"] = diffs
        
        h5.close(fid)
    end
        
    
    return
    
end
