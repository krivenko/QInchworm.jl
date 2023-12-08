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

using MPI; MPI.Init()

import PyPlot; plt = PyPlot

using Interpolations
using LinearAlgebra: diag, diagm

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators

using QInchworm.utility: ph_conj
using QInchworm.ppgf: normalize!
using QInchworm.expansion: Expansion, InteractionPair, add_corr_operators!
using QInchworm.inchworm: inchworm!, correlator_2p
using QInchworm.mpi: ismaster

include("ppsc_reference_data.jl")

# Solve non-interacting two fermion AIM coupled to
# semi-circular (Bethe lattice) hybridization functions.
#
# Performing two kinds of tests:
#
# 1. Checking that the InchWorm expansion does not break particle-hole
# symmetry for an AIM with ph-symmetry.
#
# 2. Compare to numerically exact results for the 1st, 2nd and 3rd
# order dressed self-consistent expansion for the many-body
# density matrix (computed using DLR elsewhere).
#
# Note that the 1, 2, 3 order density matrix differs from the
# exact density matrix of the non-interacting system, since
# the low order expansions introduce "artificial" effective
# interactions between hybridization insertions.

function run_dimer(nτ, orders, orders_bare, orders_gf, N_samples)

    β = 10.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0
    μ_bethe = 0.25

    # ED solution

    H_imp = -μ * (op.n(1) + op.n(2))

    # Impurity problem

    contour = kd.ImaginaryContour(β=β)
    grid = kd.ImaginaryTimeGrid(contour, nτ)

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

    # Hybridization propagator

    Δ = V^2 * kd.ImaginaryTimeGF(kd.bethe_dos(t=t_bethe/2, ϵ=μ_bethe), grid)

    # Pseudo Particle Strong Coupling Expansion

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), ph_conj(Δ))
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), ph_conj(Δ))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    inchworm!(expansion, grid, orders, orders_bare, N_samples; n_pts_after_max=1)

    normalize!(expansion.P, β)

    add_corr_operators!(expansion, (op.c(1), op.c_dag(1)))
    g = -correlator_2p(expansion, grid, orders_gf, N_samples)

    if ismaster()
        τ = kd.imagtimes(g[1].grid)
        τ_ref = collect(LinRange(0, β, 128))

        plt.figure(figsize=(3.25*4, 12))
        subp = [5, 2, 1]

        for s in 1:length(expansion.P)
            plt.subplot(subp...); subp[end] += 1;

            x = collect(τ)
            y = collect(imag(expansion.P[s].mat.data[1, 1, :]))
            P_int = interpolate((x,), y, Gridded(Linear()))
            P = P_int.(τ_ref)

            plt.plot(τ_ref, -P, label="P$(s)")
            plt.plot(τ_ref, -get_P_nca()[s], "k--", label="NCA $s ref", alpha=0.25)
            plt.plot(τ_ref, -get_P_oca()[s], "k:", label="OCA $s ref", alpha=0.25)
            plt.plot(τ_ref, -get_P_tca()[s], "k-.", label="TCA $s ref", alpha=0.25)
            plt.semilogy([], [])
            plt.ylabel(raw"$P_\Gamma(\tau)$")
            plt.xlabel(raw"$\tau$")
            plt.legend(loc="best")

            plt.subplot(subp...); subp[end] += 1;
            plt.plot(τ_ref, P - get_P_nca()[s], "k--", label="NCA $s ref", alpha=0.25)
            plt.plot(τ_ref, P - get_P_oca()[s], "k:", label="OCA $s ref", alpha=0.25)
            plt.plot(τ_ref, P - get_P_tca()[s], "k-.", label="TCA $s ref", alpha=0.25)
            plt.ylabel(raw"$\Delta P_\Gamma(\tau)$")
            plt.xlabel(raw"$\tau$")
        end

        x = collect(τ)
        y = collect(imag(g[1].mat.data[1, 1, :]))
        g_int = interpolate((x,), y, Gridded(Linear()))
        gr = g_int.(τ_ref)

        plt.subplot(subp...); subp[end] += 1;
        plt.title("nτ = $(length(τ)), N_samples = $N_samples")
        plt.plot(τ_ref, imag(get_g_nca()), label="NCA ref")
        plt.plot(τ_ref, imag(get_g_oca()), label="OCA ref")
        plt.plot(τ_ref, imag(get_g_tca()), label="TCA ref")
        plt.plot(τ, imag(g[1].mat.data[1, 1, :]), "--", label="InchW")
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$G_{11}(\tau)$")
        plt.legend(loc="best")

        plt.subplot(subp...); subp[end] += 1;
        plt.title("nτ = $(length(τ)), N_samples = $N_samples")
        plt.plot(τ_ref, abs.(gr - imag(get_g_nca())), label="NCA ref")
        plt.plot(τ_ref, abs.(gr - imag(get_g_oca())), label="OCA ref")
        plt.plot(τ_ref, abs.(gr - imag(get_g_tca())), label="TCA ref")
        plt.semilogy([], [])
        plt.xlabel(raw"$\tau$")
        plt.ylabel(raw"$\Delta G_{11}(\tau)$")
        plt.legend(loc="best")

        plt.tight_layout()
        plt.savefig("figure_ntau_$(nτ)_N_samples_$(N_samples)_orders_$(orders).pdf")
    end
end

for o = [1, 2, 3]
    orders = 0:o
    orders_gf = 0:(o-1)
    for nτ = [128*8]
        for N_samples = [8*2^7]
            run_dimer(nτ, orders, orders, orders_gf, N_samples)
        end
    end
end
