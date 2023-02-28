""" Solve non-interacting two fermion AIM coupled to
semi-circular (Behte lattice) hybridization functions.

Performing two kinds of tests:

1. Checking that the InchWorm expansion does not break particle-hole
symmetry for an AIM with ph-symmetry.

2. Compare to numerically exact results for the 1st, 2nd and 3rd
order dressed self-consistent expansion for the many-body
density matrix (computed using DLR elsewhere).

Note that the 1,2, 3 order density matrix differs from the
exact density matrix of the non-interacting system, since
the low order expansions introduce "artificial" effective
interactions between hybridization insertions.

Author: Hugo U. R. Strand (2023)

"""

using MPI; MPI.Init()

using Test

import LinearAlgebra

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.ppgf: normalize!
import QInchworm.configuration: Expansion, InteractionPair
import QInchworm.inchworm: inchworm_matsubara!, compute_gf_matsubara

import QInchworm.KeldyshED_addons: reduced_density_matrix, density_matrix
using  QInchworm.utility: inch_print

using QuadGK: quadgk

import PyPlot as plt


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

function ρ_from_n_ref(ρ_wrm, n_ref)
    ρ_ref = zero(ρ_wrm)
    ρ_ref[1, 1] = (1 - n_ref) * (1 - n_ref)
    ρ_ref[2, 2] = n_ref * (1 - n_ref)
    ρ_ref[3, 3] = n_ref * (1 - n_ref)
    ρ_ref[4, 4] = n_ref * n_ref
    return ρ_ref
end

function ρ_from_ρ_ref(ρ_wrm, ρ_ref)
    ρ = zero(ρ_wrm)
    ρ[1, 1] = ρ_ref[1]
    ρ[2, 2] = ρ_ref[2]
    ρ[3, 3] = ρ_ref[3]
    ρ[4, 4] = ρ_ref[4]
    return ρ
end

function get_ρ_exact(ρ_wrm)
    n = 0.5460872495307262 # from DLR calc
    return ρ_from_n_ref(ρ_wrm, n)
end

function get_ρ_nca(ρ_wrm)
    rho_nca = [ 0.1961713995875524, 0.2474226001525296, 0.2474226001525296, 0.3089834001073883,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_nca)
end

function get_ρ_oca(ρ_wrm)
    rho_oca = [ 0.2018070389569783, 0.2476929924482211, 0.2476929924482211, 0.3028069761465793,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_oca)
end

function get_ρ_tca(ρ_wrm)
    rho_tca = [ 0.205163794520457, 0.2478638876741985, 0.2478638876741985, 0.2991084301311462,  ]
    return ρ_from_ρ_ref(ρ_wrm , rho_tca)
end

function get_g_nca()
    g_nca = [
       -0.443594  , -0.42008018, -0.39873717, -0.37932407, -0.36163056,
       -0.34547259, -0.33068882, -0.31713748, -0.30469376, -0.29324757,
       -0.28270159, -0.27296968, -0.2639754 , -0.2556508 , -0.2479354 ,
       -0.24077525, -0.23412216, -0.22793298, -0.22216907, -0.21679569,
       -0.21178164, -0.20709881, -0.20272185, -0.19862786, -0.19479614,
       -0.19120793, -0.18784624, -0.18469562, -0.18174206, -0.17897279,
       -0.17637618, -0.17394164, -0.1716595 , -0.16952092, -0.16751785,
       -0.1656429 , -0.16388932, -0.16225094, -0.1607221 , -0.15929763,
       -0.15797279, -0.15674325, -0.15560503, -0.15455452, -0.15358841,
       -0.15270368, -0.1518976 , -0.15116766, -0.15051161, -0.14992742,
       -0.14941327, -0.14896752, -0.14858873, -0.14827563, -0.14802713,
       -0.14784228, -0.14772029, -0.14766055, -0.14766254, -0.14772594,
       -0.14785052, -0.14803622, -0.14828309, -0.14859134, -0.1489613 ,
       -0.14939342, -0.14988832, -0.15044674, -0.15106955, -0.15175779,
       -0.15251263, -0.15333539, -0.15422758, -0.15519084, -0.15622701,
       -0.15733808, -0.15852627, -0.15979397, -0.16114381, -0.16257861,
       -0.16410147, -0.16571573, -0.16742498, -0.16923314, -0.17114442,
       -0.17316337, -0.17529492, -0.17754436, -0.17991741, -0.18242027,
       -0.18505961, -0.18784262, -0.1907771 , -0.19387146, -0.19713481,
       -0.20057699, -0.20420866, -0.20804139, -0.21208769, -0.21636116,
       -0.22087657, -0.22564996, -0.2306988 , -0.2360421 , -0.24170058,
       -0.24769688, -0.2540557 , -0.26080406, -0.26797153, -0.27559054,
       -0.28369666, -0.29232897, -0.30153045, -0.31134845, -0.32183518,
       -0.33304828, -0.34505148, -0.35791536, -0.37171816, -0.38654675,
       -0.40249773, -0.41967865, -0.43820948, -0.45822421, -0.47987272,
       -0.50332296, -0.52876343, -0.556406]
    -im * g_nca
end

function get_g_oca()
    g_oca = [
       -0.44950003, -0.43140822, -0.41472244, -0.39930261, -0.38502587,
       -0.37178412, -0.359482  , -0.3480351 , -0.3373685 , -0.32741554,
       -0.31811674, -0.30941886, -0.3012742 , -0.29363986, -0.2864772 ,
       -0.27975135, -0.27343075, -0.26748681, -0.26189357, -0.25662742,
       -0.25166683, -0.24699218, -0.24258553, -0.23843046, -0.23451193,
       -0.23081613, -0.22733039, -0.22404307, -0.22094343, -0.2180216 ,
       -0.21526848, -0.21267566, -0.21023541, -0.20794055, -0.2057845 ,
       -0.20376113, -0.20186482, -0.20009035, -0.19843292, -0.19688807,
       -0.19545171, -0.19412007, -0.19288966, -0.19175728, -0.19072   ,
       -0.18977512, -0.18892019, -0.18815296, -0.1874714 , -0.18687368,
       -0.18635814, -0.18592331, -0.1855679 , -0.18529077, -0.18509095,
       -0.18496759, -0.18492004, -0.18494776, -0.18505035, -0.18522755,
       -0.18547926, -0.18580548, -0.18620636, -0.18668218, -0.18723334,
       -0.18786038, -0.18856398, -0.18934494, -0.19020419, -0.19114281,
       -0.19216202, -0.19326316, -0.19444775, -0.19571741, -0.19707396,
       -0.19851936, -0.20005573, -0.20168535, -0.2034107 , -0.20523444,
       -0.20715939, -0.2091886 , -0.21132533, -0.21357303, -0.21593542,
       -0.21841642, -0.22102024, -0.22375133, -0.22661443, -0.22961458,
       -0.23275715, -0.2360478 , -0.2394926 , -0.24309793, -0.24687062,
       -0.25081789, -0.25494741, -0.25926734, -0.26378632, -0.26851355,
       -0.2734588 , -0.27863243, -0.28404547, -0.28970965, -0.29563744,
       -0.30184209, -0.30833773, -0.31513938, -0.32226305, -0.3297258 ,
       -0.33754581, -0.34574249, -0.35433653, -0.36335004, -0.37280665,
       -0.38273162, -0.39315198, -0.40409667, -0.41559672, -0.42768539,
       -0.44039841, -0.45377416, -0.46785395, -0.48268226, -0.49830705,
       -0.51478014, -0.53215755, -0.55049997]
    -im * g_oca
end

function run_hubbard_dimer(ntau, orders, orders_bare, orders_gf, N_samples, μ_bethe)

    β = 10.0
    V = 0.5
    μ = 0.0
    t_bethe = 1.0

    # -- ED solution

    H_imp = -μ * (op.n(1) + op.n(2))
        
    # -- Impurity problem

    contour = kd.ImaginaryContour(β=β);
    grid = kd.ImaginaryTimeGrid(contour, ntau);
    
    soi = KeldyshED.Hilbert.SetOfIndices([[1], [2]])
    ed = KeldyshED.EDCore(H_imp, soi)
    
    # -- Hybridization propagator

    tau = [ real(im * τ.bpoint.val) for τ in grid ]
    delta_bethe = V^2 * semi_circular_g_tau(tau, t_bethe, μ_bethe, β)
    
    Δ = kd.ImaginaryTimeGF(
        (t1, t2) -> 1.0im * V^2 *
            semi_circular_g_tau(
                [-imag(t1.bpoint.val - t2.bpoint.val)],
                t_bethe, μ_bethe, β)[1],
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

    ip_1_fwd = InteractionPair(op.c_dag(1), op.c(1), Δ)
    ip_1_bwd = InteractionPair(op.c(1), op.c_dag(1), reverse(Δ))
    ip_2_fwd = InteractionPair(op.c_dag(2), op.c(2), Δ)
    ip_2_bwd = InteractionPair(op.c(2), op.c_dag(2), reverse(Δ))
    expansion = Expansion(ed, grid, [ip_1_fwd, ip_1_bwd, ip_2_fwd, ip_2_bwd])

    ρ_0 = density_matrix(expansion.P0, ed)
    
    inchworm_matsubara!(expansion, grid, orders, orders_bare, N_samples)

    normalize!(expansion.P, β)
    ρ_wrm = density_matrix(expansion.P, ed)

    ρ_exa = get_ρ_exact(ρ_wrm)
    ρ_nca = get_ρ_nca(ρ_wrm)
    ρ_oca = get_ρ_oca(ρ_wrm)
    ρ_tca = get_ρ_tca(ρ_wrm)
    
    diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
    diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
    diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
    diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

    ρ_000 = real(LinearAlgebra.diag(ρ_0))
    ρ_exa = real(LinearAlgebra.diag(ρ_exa))
    ρ_nca = real(LinearAlgebra.diag(ρ_nca))
    ρ_oca = real(LinearAlgebra.diag(ρ_oca))
    ρ_tca = real(LinearAlgebra.diag(ρ_tca))
    ρ_wrm = real(LinearAlgebra.diag(ρ_wrm))
    
    if inch_print()
        @show ρ_000        
        @show ρ_nca
        @show ρ_oca
        @show ρ_tca
        @show ρ_exa
        @show ρ_wrm

        @show sum(ρ_wrm)
        @show ρ_wrm[2] - ρ_wrm[3]

        @show diff_nca
        @show diff_oca
        @show diff_tca
        @show diff_exa
    end

    push!(expansion.corr_operators, (op.c(1), op.c_dag(1)))
    g = compute_gf_matsubara(expansion, grid, orders_gf, N_samples)

    diff_g_nca = maximum(abs.(get_g_nca() - g[1].mat.data[1, 1, :]))
    diff_g_oca = maximum(abs.(get_g_oca() - g[1].mat.data[1, 1, :]))

    if inch_print()
        
        @show diff_g_nca
        @show diff_g_oca

    end

    if false
        
    τ = kd.imagtimes(g[1].grid)
    τ_ref = collect(LinRange(0, β, 128))
    
    plt.figure(figsize=(3.25*2, 8))
    subp = [2, 1, 1]
    
    plt.subplot(subp...); subp[end] += 1;
    for s in 1:length(expansion.P)
        plt.plot(τ, imag(expansion.P[s].mat.data[1, 1, :]), label="P$(s)")
    end
    plt.ylabel(raw"$P_\Gamma(\tau)$")
    plt.xlabel(raw"$\tau$")
    plt.legend(loc="best")
    
    plt.subplot(subp...); subp[end] += 1;
    plt.plot(τ_ref, imag(get_g_nca()), label="NCA ref")
    plt.plot(τ_ref, imag(get_g_oca()), label="OCA ref")
    plt.plot(τ, real(g[1].mat.data[1, 1, :]), "-", label="InchW re")
    plt.plot(τ, imag(g[1].mat.data[1, 1, :]), "--", label="InchW im")
    plt.xlabel(raw"$\tau$")
    plt.ylabel(raw"$G_{11}(\tau)$")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

    end
    
    return ρ_wrm, diff_exa, diff_nca, diff_oca, diff_tca, diff_g_nca, diff_g_oca
end

@testset "bethe_order1" begin
    
    ntau = 128
    orders = 0:1
    orders_gf = 0:0
    N_samples = 8 * 2^5
    μ_bethe = 0.25
    
    ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca, diff_g_nca, diff_g_oca =
        run_hubbard_dimer(ntau, orders, orders, orders_gf, N_samples, μ_bethe)

    @test diffs_nca < 2e-3
    @test diffs_nca < diffs_oca 
    @test diffs_nca < diffs_tca
    @test diffs_nca < diffs_exa

    @test diff_g_nca < 4e-3
    @test diff_g_nca < diff_g_oca

end
