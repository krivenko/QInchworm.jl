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
# Authors: Igor Krivenko, Hugo U. R. Strand

using Test

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.utility: ph_conj
using QInchworm.spline_gf: SplineInterpolatedGF, update_interpolants!

using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.configuration: Configuration, get_diagrams_at_order
using QInchworm.diagrammatics: generate_topologies
using QInchworm; cfg = QInchworm.configuration
using QInchworm.qmc_integrate: qmc_time_ordered_integral_root

import QInchworm.ppgf

function set_matsubara!(g::kd.GenericTimeGF{T, scalar, kd.FullTimeGrid} where {T, scalar},
                        τ,
                        value)
    tau_grid = g.grid[kd.imaginary_branch]

    τ_0 = tau_grid[1]
    τ_beta = tau_grid[end]

    sidx = τ.cidx
    eidx = τ_beta.cidx

    for τ_1 in g.grid[sidx:eidx]
        i1 = τ_1.cidx
        i2 = τ_0.cidx + τ_1.cidx - τ.cidx
        t1 = g.grid[i1]
        t2 = g.grid[i2]
        g[t1, t2] = value
    end
end

function set_matsubara!(g::kd.ImaginaryTimeGF{T, scalar}, τ, value) where {T, scalar}
    tau_grid = g.grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    g[τ, τ_0] = value
end

function set_matsubara!(
    g::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar} where {T, scalar},
    τ, value)
    g[τ, g.grid[1], τ_max=τ] = value
end

function ppgf.normalize!(
    g::SplineInterpolatedGF{kd.ImaginaryTimeGF{T, scalar}, T, scalar} where {T, scalar},
    λ)
    τ_0 = g.grid[1]
    for τ in g.grid
        g.GF[τ, τ_0] = g[τ, τ_0] .* exp(-1im * τ.bpoint.val * λ)
    end
    update_interpolants!(g)
end

@testset "Equilibrium NCA" verbose=true begin

    # Single state
    β = 10.
    μ = +0.1 # Chemical potential
    ϵ = +0.1 # Bath energy level
    V = -0.1 # Hybridization

    # Discretization
    tmax = 1.
    nt = 10
    nτ= 30

    # Quasi Monte Carlo
    N = 2^13

    # Exact Diagonalization

    H = - μ * op.n("0")
    soi = ked.Hilbert.SetOfIndices([["0"]]);
    ed = ked.EDCore(H, soi)
    ρ = ked.density_matrix(ed, β)

    """1st order inching on the imaginary branch: Riemann sum integration"""
    function run_nca_equil_tests_riemann(contour, grid, Δ, interpolate_ppgf=false)

        ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
        ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), ph_conj(Δ))
        ppsc_exp = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=interpolate_ppgf)

        tau_grid = grid[kd.imaginary_branch]
        τ_0, τ_beta = tau_grid[1], tau_grid[end]
        Δτ = -imag(step(grid, kd.imaginary_branch))

        # 0-order configuration
        diag_0 = get_diagrams_at_order(ppsc_exp, generate_topologies(0), 0)[1]
        conf_0 = Configuration(diag_0, ppsc_exp, 0)
        cfg.set_initial_node_time!(conf_0, τ_0.bpoint)

        # 1-st order configurations
        diags_1 = get_diagrams_at_order(ppsc_exp, generate_topologies(1), 1)
        confs_1 = Configuration.(diags_1, Ref(ppsc_exp), 1)
        cfg.set_initial_node_time!.(confs_1, Ref(τ_0.bpoint))

        for (fidx, τ_f) in enumerate(tau_grid[2:end])

            τ_w = tau_grid[fidx]

            cfg.set_inchworm_node_time!(conf_0, τ_w.bpoint)
            cfg.set_final_node_time!(conf_0, τ_f.bpoint)

            val = cfg.eval(ppsc_exp, conf_0)

            cfg.set_inchworm_node_time!.(confs_1, Ref(τ_w.bpoint))
            cfg.set_final_node_time!.(confs_1, Ref(τ_f.bpoint))

            for τ_1 in tau_grid[1:fidx]
                pair_node_times = [τ_f.bpoint, τ_1.bpoint]
                cfg.update_pair_node_times!.(confs_1, diags_1, Ref(pair_node_times))
                val -= Δτ^2 * cfg.eval(ppsc_exp, confs_1[1])
                val -= Δτ^2 * cfg.eval(ppsc_exp, confs_1[2])
            end

            for (s, P_s) in enumerate(ppsc_exp.P)
                sf, mat = val[s]
                set_matsubara!(P_s, τ_f, mat)
            end

        end

        ppgf.normalize!(ppsc_exp.P, β)

        Z = ppgf.partition_function(ppsc_exp.P)

        # Regression tests
        @test Z ≈ 1.0
        @test ppsc_exp.P[1][τ_beta, τ_0] + ppsc_exp.P[2][τ_beta, τ_0] ≈ [-im]
        @test ppsc_exp.P[1][τ_beta, τ_0][1, 1] ≈ 0.0 - 0.7127769872093338im
        @test ppsc_exp.P[2][τ_beta, τ_0][1, 1] ≈ 0.0 - 0.2872230127906664im

        interpolate_ppgf && return

        # Single particle Green's function
        g = ppgf.first_order_spgf(ppsc_exp.P, ed, 1, 1)

        #
        # Reference calculation of the SPGF
        #

        H_ref = - μ * op.n("0") + ϵ * op.n("1") +
                V * op.c_dag("1") * op.c("0") + V * op.c_dag("0") * op.c("1")
        soi_ref = ked.Hilbert.SetOfIndices([["0"], ["1"]]);
        ed_ref = ked.EDCore(H_ref, soi_ref)

        idx = ked.Hilbert.IndicesType(["0"])
        g_ref = ked.computegf(ed_ref, grid, idx, idx);

        @test isapprox(g[:matsubara], g_ref[:matsubara], atol=0.05)
    end

    """1st order inching on the imaginary branch: Quasi Monte Carlo integration"""
    function run_nca_equil_tests_qmc(contour, grid, Δ, interpolate_ppgf=false)

        ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
        ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), ph_conj(Δ))
        ppsc_exp = Expansion(ed, grid, [ip_fwd, ip_bwd], interpolate_ppgf=interpolate_ppgf)

        tau_grid = grid[kd.imaginary_branch]
        τ_0, τ_beta = tau_grid[1], tau_grid[end]

        # 0-order configuration
        diag_0 = get_diagrams_at_order(ppsc_exp, generate_topologies(0), 0)[1]
        conf_0 = Configuration(diag_0, ppsc_exp, 0)
        cfg.set_initial_node_time!(conf_0, τ_0.bpoint)

        # 1-st order configurations
        diags_1 = get_diagrams_at_order(ppsc_exp, generate_topologies(1), 1)
        confs_1 = Configuration.(diags_1, Ref(ppsc_exp), 1)
        cfg.set_initial_node_time!.(confs_1, Ref(τ_0.bpoint))

        for (fidx, τ_f) in enumerate(tau_grid[2:end])

            τ_w = tau_grid[fidx]

            cfg.set_inchworm_node_time!(conf_0, τ_w.bpoint)
            cfg.set_final_node_time!(conf_0, τ_f.bpoint)

            val = cfg.eval(ppsc_exp, conf_0)

            cfg.set_inchworm_node_time!.(confs_1, Ref(τ_w.bpoint))
            cfg.set_final_node_time!.(confs_1, Ref(τ_f.bpoint))

            # The 'im' prefactor accounts for the direction of the imaginary branch
            val -= im * qmc_time_ordered_integral_root(
                1,
                contour,
                τ_0.bpoint,
                τ_w.bpoint,
                init = zero(val),
                N = N) do τ_1
                    pair_node_times = [τ_f.bpoint, τ_1[1]]
                    cfg.update_pair_node_times!.(confs_1, diags_1, Ref(pair_node_times))
                    cfg.eval(ppsc_exp, confs_1[1]) + cfg.eval(ppsc_exp, confs_1[2])
                end

            for (s, P_s) in enumerate(ppsc_exp.P)
                sf, mat = val[s]
                set_matsubara!(P_s, τ_f, mat)
            end

        end

        ppgf.normalize!(ppsc_exp.P, β)

        # ---------------------
        # -- Regression test --
        # ---------------------

        @test ppsc_exp.P[1][τ_beta, τ_0] + ppsc_exp.P[2][τ_beta, τ_0] ≈ [-im]
        @test isapprox(ppsc_exp.P[1][τ_beta, τ_0][1, 1],
                       0.0 - 0.6821011169782484im,
                       atol=1e-3)
        @test isapprox(ppsc_exp.P[2][τ_beta, τ_0][1, 1],
                       0.0 - 0.3178988830217517im,
                       atol=1e-3)
    end

    @testset "Imaginary time" begin
        contour = kd.ImaginaryContour(β=β)
        grid = kd.ImaginaryTimeGrid(contour, nτ)

        # Hybridization propagator
        Δ = kd.ImaginaryTimeGF(kd.DeltaDOS([ϵ], [V^2]), grid)

        run_nca_equil_tests_riemann(contour, grid, Δ, false)
        run_nca_equil_tests_qmc(contour, grid, Δ, false)

        #FIXME: when going to higher order interpolants
        @test_skip run_nca_equil_tests_riemann(contour, grid, SplineInterpolatedGF(Δ), true)
        @test_skip run_nca_equil_tests_qmc(contour, grid, SplineInterpolatedGF(Δ), true)
    end

    #FIXME when generalizing to real time
    @test_skip @testset "Twisted Kadanoff-Baym-Keldysh contour" begin
        contour = kd.twist(kd.FullContour(tmax=tmax, β=β))
        grid = kd.FullTimeGrid(contour, nt, nτ)

        # Hybridization propagator
        Δ = kd.FullTimeGF(kd.DeltaDOS([ϵ], [V^2]), grid)

        @test_skip run_nca_equil_tests_riemann(contour, grid, Δ)
        @test_skip run_nca_equil_tests_qmc(contour, grid, Δ)
    end

end
