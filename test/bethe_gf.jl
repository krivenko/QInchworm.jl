""" Solve non-interacting two fermion AIM coupled to
semi-circular (Bethe lattice) hybridization functions.

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

#import PyPlot; plt = PyPlot

using LinearAlgebra: diag
using QuadGK: quadgk

using Keldysh; kd = Keldysh
using KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

using QInchworm.ppgf: normalize!, density_matrix
using QInchworm.expansion: Expansion, InteractionPair
using QInchworm.inchworm: inchworm_matsubara!, compute_gf_matsubara
using QInchworm.utility: inch_print

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

function get_G_tca()
G0_j = [-1.        , -0.94903547, -0.90161241, -0.85743829, -0.8162487 ,
       -0.7778043 , -0.74188812, -0.70830315, -0.67687026, -0.64742631,
       -0.61982259, -0.59392328, -0.56960425, -0.54675184, -0.5252619 ,
       -0.50503887, -0.48599496, -0.46804942, -0.45112795, -0.43516204,
       -0.4200885 , -0.405849  , -0.3923896 , -0.37966043, -0.36761529,
       -0.35621138, -0.34540901, -0.33517132, -0.3254641 , -0.31625554,
       -0.30751603, -0.29921803, -0.29133589, -0.28384569, -0.27672511,
       -0.26995336, -0.26351099, -0.25737985, -0.25154297, -0.24598447,
       -0.24068952, -0.2356442 , -0.23083552, -0.22625126, -0.22188001,
       -0.21771106, -0.21373435, -0.20994046, -0.20632054, -0.2028663 ,
       -0.19956994, -0.19642413, -0.19342202, -0.19055713, -0.18782341,
       -0.18521515, -0.18272701, -0.18035396, -0.17809127, -0.17593453,
       -0.17387955, -0.17192244, -0.17005954, -0.1682874 , -0.16660282,
       -0.16500278, -0.16348446, -0.16204524, -0.16068266, -0.15939443,
       -0.15817843, -0.15703269, -0.15595537, -0.15494482, -0.15399946,
       -0.1531179 , -0.15229885, -0.15154115, -0.15084375, -0.15020573,
       -0.14962629, -0.14910473, -0.14864047, -0.14823303, -0.14788205,
       -0.14758729, -0.14734859, -0.14716594, -0.1470394 , -0.14696917,
       -0.14695557, -0.146999  , -0.14710003, -0.14725932, -0.14747768,
       -0.14775603, -0.14809545, -0.14849715, -0.1489625 , -0.14949304,
       -0.15009046, -0.15075663, -0.15149363, -0.15230372, -0.15318937,
       -0.15415331, -0.15519849, -0.15632812, -0.15754571, -0.15885507,
       -0.16026033, -0.16176598, -0.1633769 , -0.16509838, -0.16693617,
       -0.16889652, -0.1709862 , -0.17321257, -0.17558364, -0.17810812,
       -0.18079548, -0.18365604, -0.18670104, -0.18994276, -0.19339457,
       -0.19707109, -0.20098833, -0.20516379]
G1_j = [-1.        , -0.94925819, -0.90244124, -0.85917531, -0.81912843,
       -0.78200518, -0.74754214, -0.71550398, -0.68568001, -0.65788127,
       -0.63193795, -0.6076972 , -0.58502112, -0.56378516, -0.54387656,
       -0.52519314, -0.50764206, -0.49113894, -0.47560688, -0.46097573,
       -0.44718144, -0.43416536, -0.42187381, -0.41025751, -0.39927124,
       -0.38887337, -0.37902561, -0.36969266, -0.36084193, -0.35244334,
       -0.34446906, -0.33689335, -0.32969234, -0.32284391, -0.31632755,
       -0.31012418, -0.30421608, -0.29858675, -0.29322086, -0.2881041 ,
       -0.28322314, -0.27856554, -0.2741197 , -0.26987478, -0.26582064,
       -0.26194783, -0.25824748, -0.2547113 , -0.25133154, -0.24810093,
       -0.24501265, -0.24206034, -0.23923801, -0.23654005, -0.23396121,
       -0.23149656, -0.22914148, -0.22689162, -0.22474293, -0.22269159,
       -0.22073402, -0.21886687, -0.21708701, -0.2153915 , -0.21377759,
       -0.2122427 , -0.21078444, -0.20940056, -0.20808897, -0.20684774,
       -0.20567506, -0.20456925, -0.20352877, -0.20255221, -0.20163825,
       -0.20078569, -0.19999347, -0.1992606 , -0.1985862 , -0.1979695 ,
       -0.19740983, -0.19690659, -0.19645931, -0.19606758, -0.19573112,
       -0.1954497 , -0.1952232 , -0.19505161, -0.19493498, -0.19487346,
       -0.19486732, -0.1949169 , -0.19502263, -0.19518507, -0.19540485,
       -0.19568272, -0.19601954, -0.19641629, -0.19687405, -0.19739402,
       -0.19797755, -0.19862611, -0.1993413 , -0.20012489, -0.2009788 ,
       -0.20190509, -0.20290604, -0.20398409, -0.20514187, -0.20638225,
       -0.2077083 , -0.20912334, -0.21063096, -0.21223502, -0.21393967,
       -0.2157494 , -0.21766903, -0.21970377, -0.22185921, -0.22414141,
       -0.22655688, -0.22911265, -0.23181631, -0.23467603, -0.23770069,
       -0.24089984, -0.24428384, -0.24786389]
G2_j = [-1.        , -0.94925819, -0.90244124, -0.85917531, -0.81912843,
       -0.78200518, -0.74754214, -0.71550398, -0.68568001, -0.65788127,
       -0.63193795, -0.6076972 , -0.58502112, -0.56378516, -0.54387656,
       -0.52519314, -0.50764206, -0.49113894, -0.47560688, -0.46097573,
       -0.44718144, -0.43416536, -0.42187381, -0.41025751, -0.39927124,
       -0.38887337, -0.37902561, -0.36969266, -0.36084193, -0.35244334,
       -0.34446906, -0.33689335, -0.32969234, -0.32284391, -0.31632755,
       -0.31012418, -0.30421608, -0.29858675, -0.29322086, -0.2881041 ,
       -0.28322314, -0.27856554, -0.2741197 , -0.26987478, -0.26582064,
       -0.26194783, -0.25824748, -0.2547113 , -0.25133154, -0.24810093,
       -0.24501265, -0.24206034, -0.23923801, -0.23654005, -0.23396121,
       -0.23149656, -0.22914148, -0.22689162, -0.22474293, -0.22269159,
       -0.22073402, -0.21886687, -0.21708701, -0.2153915 , -0.21377759,
       -0.2122427 , -0.21078444, -0.20940056, -0.20808897, -0.20684774,
       -0.20567506, -0.20456925, -0.20352877, -0.20255221, -0.20163825,
       -0.20078569, -0.19999347, -0.1992606 , -0.1985862 , -0.1979695 ,
       -0.19740983, -0.19690659, -0.19645931, -0.19606758, -0.19573112,
       -0.1954497 , -0.1952232 , -0.19505161, -0.19493498, -0.19487346,
       -0.19486732, -0.1949169 , -0.19502263, -0.19518507, -0.19540485,
       -0.19568272, -0.19601954, -0.19641629, -0.19687405, -0.19739402,
       -0.19797755, -0.19862611, -0.1993413 , -0.20012489, -0.2009788 ,
       -0.20190509, -0.20290604, -0.20398409, -0.20514187, -0.20638225,
       -0.2077083 , -0.20912334, -0.21063096, -0.21223502, -0.21393967,
       -0.2157494 , -0.21766903, -0.21970377, -0.22185921, -0.22414141,
       -0.22655688, -0.22911265, -0.23181631, -0.23467603, -0.23770069,
       -0.24089984, -0.24428384, -0.24786389]
G3_j = [-1.        , -0.94948098, -0.90327107, -0.86091699, -0.82202156,
       -0.78623586, -0.75325256, -0.72280023, -0.6946386 , -0.66855441,
       -0.64435794, -0.62188003, -0.60096951, -0.58149098, -0.56332292,
       -0.54635606, -0.53049192, -0.51564161, -0.50172475, -0.48866851,
       -0.47640681, -0.4648796 , -0.4540322 , -0.4438148 , -0.43418192,
       -0.42509198, -0.41650695, -0.40839198, -0.4007151 , -0.39344695,
       -0.38656054, -0.38003103, -0.37383556, -0.36795303, -0.36236398,
       -0.35705045, -0.35199581, -0.34718469, -0.34260288, -0.33823719,
       -0.33407541, -0.33010619, -0.32631901, -0.32270411, -0.31925239,
       -0.31595542, -0.31280532, -0.30979479, -0.30691701, -0.30416564,
       -0.30153475, -0.29901882, -0.29661272, -0.29431163, -0.29211108,
       -0.29000688, -0.28799512, -0.28607216, -0.28423459, -0.28247923,
       -0.2808031 , -0.27920343, -0.27767764, -0.2762233 , -0.27483817,
       -0.27352013, -0.27226725, -0.27107768, -0.26994975, -0.26888187,
       -0.2678726 , -0.26692057, -0.26602454, -0.26518337, -0.26439599,
       -0.26366145, -0.26297886, -0.26234742, -0.26176642, -0.26123521,
       -0.26075322, -0.26031995, -0.25993499, -0.25959795, -0.25930856,
       -0.25906658, -0.25887185, -0.25872426, -0.25862377, -0.25857041,
       -0.25856425, -0.25860546, -0.25869423, -0.25883084, -0.25901563,
       -0.25924899, -0.25953139, -0.25986336, -0.2602455 , -0.26067848,
       -0.26116305, -0.26170003, -0.2622903 , -0.26293486, -0.26363474,
       -0.26439111, -0.26520519, -0.26607832, -0.26701193, -0.26800756,
       -0.26906685, -0.27019157, -0.27138361, -0.27264498, -0.27397783,
       -0.27538448, -0.27686739, -0.27842917, -0.28007264, -0.28180078,
       -0.28361681, -0.28552413, -0.2875264 , -0.28962751, -0.29183165,
        -0.29414327, -0.29656716, -0.29910843]
    return G0_j, G1_j, G2_j, G3_j
end

function get_G_oca()
    G0_j = [-1.        , -0.95201873, -0.90728941, -0.86554819, -0.82655597,
       -0.79009574, -0.75597032, -0.72400027, -0.69402216, -0.66588687,
       -0.63945824, -0.6146118 , -0.59123358, -0.56921916, -0.54847275,
       -0.52890637, -0.51043918, -0.49299674, -0.47651052, -0.46091733,
       -0.44615885, -0.43218121, -0.41893463, -0.40637301, -0.3944537 ,
       -0.38313715, -0.37238666, -0.3621682 , -0.35245013, -0.34320304,
       -0.33439957, -0.32601425, -0.31802334, -0.31040472, -0.30313777,
       -0.2962032 , -0.28958303, -0.28326045, -0.27721973, -0.27144615,
       -0.26592596, -0.26064624, -0.25559491, -0.25076064, -0.2461328 ,
       -0.24170141, -0.23745712, -0.23339114, -0.22949519, -0.22576153,
       -0.22218285, -0.21875228, -0.21546339, -0.21231009, -0.20928667,
       -0.20638775, -0.20360828, -0.20094349, -0.1983889 , -0.19594029,
       -0.19359368, -0.19134535, -0.18919178, -0.18712967, -0.18515591,
       -0.18326758, -0.18146194, -0.17973644, -0.17808866, -0.17651635,
       -0.17501741, -0.17358988, -0.17223193, -0.17094187, -0.16971812,
       -0.16855923, -0.16746387, -0.16643082, -0.16545898, -0.16454734,
       -0.163695  , -0.16290117, -0.16216516, -0.16148638, -0.16086435,
       -0.16029866, -0.15978903, -0.15933526, -0.15893726, -0.15859505,
       -0.15830872, -0.15807851, -0.15790472, -0.1577878 , -0.15772829,
       -0.15772685, -0.15778427, -0.15790145, -0.15807944, -0.15831942,
       -0.1586227 , -0.15899078, -0.15942527, -0.15992801, -0.16050097,
       -0.16114634, -0.16186651, -0.16266409, -0.16354193, -0.16450313,
       -0.16555106, -0.16668938, -0.16792209, -0.16925351, -0.17068833,
       -0.17223166, -0.17388903, -0.17566645, -0.17757046, -0.17960815,
       -0.18178723, -0.18411609, -0.18660385, -0.18926045, -0.19209671,
       -0.19512444, -0.19835653, -0.20180704]
    G1_j = [-1.        , -0.95224213, -0.90812316, -0.8673003 , -0.82946814,
       -0.79435418, -0.76171479, -0.73133188, -0.70300984, -0.67657299,
       -0.65186326, -0.62873824, -0.60706941, -0.58674061, -0.5676467 ,
       -0.54969238, -0.53279115, -0.51686435, -0.5018404 , -0.48765404,
       -0.47424571, -0.46156096, -0.44954999, -0.43816713, -0.42737052,
       -0.41712168, -0.40738525, -0.39812865, -0.38932189, -0.38093725,
       -0.37294917, -0.36533399, -0.35806982, -0.35113638, -0.34451487,
       -0.33818781, -0.33213901, -0.32635336, -0.32081685, -0.31551639,
       -0.31043981, -0.30557572, -0.30091354, -0.29644333, -0.29215583,
       -0.28804238, -0.28409486, -0.28030567, -0.27666769, -0.27317425,
       -0.26981908, -0.26659631, -0.26350043, -0.26052625, -0.25766891,
       -0.25492384, -0.25228674, -0.24975358, -0.24732056, -0.2449841 ,
       -0.24274087, -0.24058769, -0.23852162, -0.23653987, -0.23463983,
       -0.23281904, -0.23107521, -0.22940618, -0.22780993, -0.22628458,
       -0.22482837, -0.22343965, -0.22211691, -0.22085872, -0.21966377,
       -0.21853086, -0.21745889, -0.21644682, -0.21549375, -0.21459885,
       -0.21376138, -0.21298067, -0.21225618, -0.2115874 , -0.21097393,
       -0.21041547, -0.20991178, -0.2094627 , -0.20906816, -0.20872818,
       -0.20844284, -0.20821235, -0.20803696, -0.20791703, -0.20785301,
       -0.20784545, -0.20789497, -0.20800233, -0.20816835, -0.20839399,
       -0.20868031, -0.20902847, -0.20943979, -0.20991568, -0.21045772,
       -0.2110676 , -0.21174718, -0.21249848, -0.21332368, -0.21422517,
       -0.21520548, -0.2162674 , -0.21741391, -0.21864823, -0.21997384,
       -0.22139447, -0.22291416, -0.22453727, -0.22626847, -0.22811282,
       -0.23007577, -0.23216317, -0.23438135, -0.23673715, -0.23923794,
       -0.24189166, -0.24470691, -0.24769299]
    G2_j = [-1.        , -0.95224213, -0.90812316, -0.8673003 , -0.82946814,
       -0.79435418, -0.76171479, -0.73133188, -0.70300984, -0.67657299,
       -0.65186326, -0.62873824, -0.60706941, -0.58674061, -0.5676467 ,
       -0.54969238, -0.53279115, -0.51686435, -0.5018404 , -0.48765404,
       -0.47424571, -0.46156096, -0.44954999, -0.43816713, -0.42737052,
       -0.41712168, -0.40738525, -0.39812865, -0.38932189, -0.38093725,
       -0.37294917, -0.36533399, -0.35806982, -0.35113638, -0.34451487,
       -0.33818781, -0.33213901, -0.32635336, -0.32081685, -0.31551639,
       -0.31043981, -0.30557572, -0.30091354, -0.29644333, -0.29215583,
       -0.28804238, -0.28409486, -0.28030567, -0.27666769, -0.27317425,
       -0.26981908, -0.26659631, -0.26350043, -0.26052625, -0.25766891,
       -0.25492384, -0.25228674, -0.24975358, -0.24732056, -0.2449841 ,
       -0.24274087, -0.24058769, -0.23852162, -0.23653987, -0.23463983,
       -0.23281904, -0.23107521, -0.22940618, -0.22780993, -0.22628458,
       -0.22482837, -0.22343965, -0.22211691, -0.22085872, -0.21966377,
       -0.21853086, -0.21745889, -0.21644682, -0.21549375, -0.21459885,
       -0.21376138, -0.21298067, -0.21225618, -0.2115874 , -0.21097393,
       -0.21041547, -0.20991178, -0.2094627 , -0.20906816, -0.20872818,
       -0.20844284, -0.20821235, -0.20803696, -0.20791703, -0.20785301,
       -0.20784545, -0.20789497, -0.20800233, -0.20816835, -0.20839399,
       -0.20868031, -0.20902847, -0.20943979, -0.20991568, -0.21045772,
       -0.2110676 , -0.21174718, -0.21249848, -0.21332368, -0.21422517,
       -0.21520548, -0.2162674 , -0.21741391, -0.21864823, -0.21997384,
       -0.22139447, -0.22291416, -0.22453727, -0.22626847, -0.22811282,
       -0.23007577, -0.23216317, -0.23438135, -0.23673715, -0.23923794,
       -0.24189166, -0.24470691, -0.24769299]
    G3_j = [-1.        , -0.95246558, -0.90895768, -0.86905594, -0.83239057,
       -0.79863556, -0.76750289, -0.73873768, -0.71211381, -0.68743041,
       -0.66450861, -0.64318887, -0.62332865, -0.60480033, -0.58748947,
       -0.57129329, -0.5561193 , -0.54188414, -0.52851257, -0.51593657,
       -0.50409453, -0.49293061, -0.48239409, -0.47243883, -0.46302284,
       -0.45410781, -0.44565877, -0.43764376, -0.4300335 , -0.42280118,
       -0.41592218, -0.40937391, -0.40313557, -0.39718803, -0.39151365,
       -0.38609616, -0.38092054, -0.3759729 , -0.37124038, -0.36671109,
       -0.362374  , -0.35821887, -0.3542362 , -0.35041717, -0.34675356,
       -0.34323773, -0.33986258, -0.33662146, -0.33350819, -0.33051699,
       -0.32764249, -0.32487965, -0.32222376, -0.31967042, -0.31721551,
       -0.31485518, -0.31258583, -0.31040407, -0.30830674, -0.30629088,
       -0.3043537 , -0.3024926 , -0.30070515, -0.29898905, -0.29734217,
       -0.29576249, -0.29424816, -0.2927974 , -0.29140857, -0.29008015,
       -0.28881071, -0.2875989 , -0.2864435 , -0.28534334, -0.28429737,
       -0.28330459, -0.2823641 , -0.28147506, -0.28063671, -0.27984836,
       -0.27910937, -0.27841919, -0.27777732, -0.27718332, -0.27663681,
       -0.27613748, -0.27568506, -0.27527936, -0.27492022, -0.27460755,
       -0.27434133, -0.27412158, -0.27394837, -0.27382183, -0.27374217,
       -0.27370962, -0.2737245 , -0.27378716, -0.27389805, -0.27405764,
       -0.27426649, -0.27452522, -0.27483451, -0.27519511, -0.27560785,
       -0.27607364, -0.27659345, -0.27716834, -0.27779946, -0.27848804,
       -0.27923542, -0.28004302, -0.28091238, -0.28184513, -0.28284303,
       -0.28390797, -0.28504195, -0.28624713, -0.2875258 , -0.28888041,
       -0.29031359, -0.29182813, -0.29342704, -0.2951135 , -0.29689094,
        -0.29876302, -0.30073363, -0.30280698]
    return G0_j, G1_j, G2_j, G3_j
end


function get_G_nca()
    G0_j = [
       -1.        , -0.95576893, -0.9144512 , -0.87581573, -0.83965236,
       -0.80576973, -0.77399345, -0.74416436, -0.71613713, -0.68977884,
       -0.66496787, -0.64159278, -0.61955139, -0.59874992, -0.57910219,
       -0.56052897, -0.54295734, -0.52632012, -0.5105554 , -0.49560603,
       -0.48141924, -0.46794626, -0.45514198, -0.44296466, -0.43137561,
       -0.42033897, -0.40982148, -0.39979225, -0.39022259, -0.38108582,
       -0.37235712, -0.36401338, -0.35603308, -0.34839615, -0.34108386,
       -0.33407876, -0.32736451, -0.32092587, -0.31474859, -0.30881931,
       -0.30312555, -0.2976556 , -0.29239852, -0.287344  , -0.28248242,
       -0.27780472, -0.2733024 , -0.26896747, -0.26479242, -0.26077021,
       -0.25689418, -0.25315811, -0.24955609, -0.2460826 , -0.24273242,
       -0.23950064, -0.23638262, -0.233374  , -0.23047067, -0.22766874,
       -0.22496456, -0.22235468, -0.21983584, -0.21740498, -0.21505922,
       -0.21279581, -0.21061221, -0.208506  , -0.20647491, -0.20451679,
       -0.20262964, -0.20081158, -0.19906085, -0.19737579, -0.19575485,
       -0.19419661, -0.19269972, -0.19126294, -0.18988512, -0.18856522,
       -0.18730227, -0.1860954 , -0.1849438 , -0.18384677, -0.1828037 ,
       -0.18181403, -0.18087731, -0.17999315, -0.17916126, -0.17838143,
       -0.17765351, -0.17697745, -0.17635329, -0.17578114, -0.17526122,
       -0.1747938 , -0.1743793 , -0.17401817, -0.17371102, -0.17345853,
       -0.17326148, -0.17312078, -0.17303747, -0.17301268, -0.17304769,
       -0.17314392, -0.17330294, -0.17352646, -0.17381636, -0.17417471,
       -0.17460376, -0.17510596, -0.17568398, -0.17634071, -0.17707932,
       -0.17790323, -0.17881614, -0.17982208, -0.18092544, -0.18213093,
       -0.18344372, -0.18486937, -0.18641394, -0.188084  , -0.18988667,
       -0.19182973, -0.19392158, -0.1961714 ]
    G1_j = [
       -1.        , -0.95599319, -0.91529125, -0.87758725, -0.84260668,
       -0.81010364, -0.77985761, -0.75167055, -0.72536445, -0.70077901,
       -0.6777698 , -0.65620644, -0.63597114, -0.61695737, -0.5990686 ,
       -0.58221732, -0.56632408, -0.55131666, -0.53712934, -0.52370225,
       -0.51098077, -0.49891504, -0.48745946, -0.47657231, -0.46621537,
       -0.45635355, -0.44695468, -0.43798914, -0.42942972, -0.42125132,
       -0.4134308 , -0.40594683, -0.39877967, -0.39191108, -0.38532415,
       -0.37900325, -0.37293384, -0.36710245, -0.36149656, -0.35610451,
       -0.35091546, -0.3459193 , -0.34110662, -0.33646861, -0.33199708,
       -0.32768434, -0.32352322, -0.319507  , -0.31562938, -0.31188448,
       -0.30826675, -0.30477101, -0.30139239, -0.29812631, -0.29496848,
       -0.29191484, -0.28896159, -0.28610516, -0.28334217, -0.28066947,
       -0.27808405, -0.27558312, -0.27316403, -0.27082429, -0.26856154,
       -0.26637359, -0.26425836, -0.26221389, -0.26023835, -0.25833001,
       -0.25648726, -0.25470859, -0.25299257, -0.25133787, -0.24974327,
       -0.24820762, -0.24672984, -0.24530895, -0.24394404, -0.24263429,
       -0.24137893, -0.24017727, -0.23902871, -0.23793268, -0.23688873,
       -0.23589643, -0.23495544, -0.2340655 , -0.23322639, -0.23243797,
       -0.23170017, -0.23101298, -0.23037648, -0.22979079, -0.22925613,
       -0.22877277, -0.22834108, -0.22796148, -0.2276345 , -0.22736072,
       -0.22714085, -0.22697564, -0.22686596, -0.2268128 , -0.2268172 ,
       -0.22688035, -0.22700354, -0.22718818, -0.22743581, -0.2277481 ,
       -0.22812686, -0.22857406, -0.22909183, -0.22968247, -0.23034846,
       -0.23109248, -0.23191743, -0.23282641, -0.23382278, -0.23491017,
       -0.23609245, -0.23737384, -0.23875885, -0.24025236, -0.2418596 ,
       -0.24358624, -0.24543838, -0.2474226 ]
    G2_j = [
       -1.        , -0.95599319, -0.91529125, -0.87758725, -0.84260668,
       -0.81010364, -0.77985761, -0.75167055, -0.72536445, -0.70077901,
       -0.6777698 , -0.65620644, -0.63597114, -0.61695737, -0.5990686 ,
       -0.58221732, -0.56632408, -0.55131666, -0.53712934, -0.52370225,
       -0.51098077, -0.49891504, -0.48745946, -0.47657231, -0.46621537,
       -0.45635355, -0.44695468, -0.43798914, -0.42942972, -0.42125132,
       -0.4134308 , -0.40594683, -0.39877967, -0.39191108, -0.38532415,
       -0.37900325, -0.37293384, -0.36710245, -0.36149656, -0.35610451,
       -0.35091546, -0.3459193 , -0.34110662, -0.33646861, -0.33199708,
       -0.32768434, -0.32352322, -0.319507  , -0.31562938, -0.31188448,
       -0.30826675, -0.30477101, -0.30139239, -0.29812631, -0.29496848,
       -0.29191484, -0.28896159, -0.28610516, -0.28334217, -0.28066947,
       -0.27808405, -0.27558312, -0.27316403, -0.27082429, -0.26856154,
       -0.26637359, -0.26425836, -0.26221389, -0.26023835, -0.25833001,
       -0.25648726, -0.25470859, -0.25299257, -0.25133787, -0.24974327,
       -0.24820762, -0.24672984, -0.24530895, -0.24394404, -0.24263429,
       -0.24137893, -0.24017727, -0.23902871, -0.23793268, -0.23688873,
       -0.23589643, -0.23495544, -0.2340655 , -0.23322639, -0.23243797,
       -0.23170017, -0.23101298, -0.23037648, -0.22979079, -0.22925613,
       -0.22877277, -0.22834108, -0.22796148, -0.2276345 , -0.22736072,
       -0.22714085, -0.22697564, -0.22686596, -0.2268128 , -0.2268172 ,
       -0.22688035, -0.22700354, -0.22718818, -0.22743581, -0.2277481 ,
       -0.22812686, -0.22857406, -0.22909183, -0.22968247, -0.23034846,
       -0.23109248, -0.23191743, -0.23282641, -0.23382278, -0.23491017,
       -0.23609245, -0.23737384, -0.23875885, -0.24025236, -0.2418596 ,
       -0.24358624, -0.24543838, -0.2474226 ]
    G3_j = [
       -1.        , -0.95621749, -0.91613181, -0.8793612 , -0.84556807,
       -0.81445347, -0.78575226, -0.759229  , -0.73467427, -0.71190162,
       -0.69074474, -0.67105515, -0.65270006, -0.63556059, -0.61953014,
       -0.60451303, -0.59042324, -0.57718334, -0.56472357, -0.55298096,
       -0.54189862, -0.53142507, -0.52151369, -0.51212221, -0.50321222,
       -0.49474882, -0.48670023, -0.47903747, -0.47173414, -0.46476607,
       -0.45811119, -0.45174928, -0.44566179, -0.4398317 , -0.43424339,
       -0.42888247, -0.42373568, -0.41879082, -0.41403662, -0.40946266,
       -0.40505932, -0.40081767, -0.39672945, -0.392787  , -0.38898321,
       -0.38531145, -0.38176557, -0.37833986, -0.37502896, -0.37182789,
       -0.36873203, -0.36573701, -0.36283879, -0.36003356, -0.35731779,
       -0.35468813, -0.35214148, -0.34967491, -0.34728568, -0.34497121,
       -0.34272909, -0.34055705, -0.33845296, -0.33641482, -0.33444075,
       -0.33252899, -0.33067786, -0.32888583, -0.32715141, -0.32547324,
       -0.32385004, -0.32228059, -0.32076376, -0.3192985 , -0.31788383,
       -0.31651881, -0.31520259, -0.31393439, -0.31271345, -0.3115391 ,
       -0.31041072, -0.30932772, -0.3082896 , -0.30729588, -0.30634614,
       -0.30544   , -0.30457714, -0.30375727, -0.30298016, -0.30224562,
       -0.30155349, -0.30090369, -0.30029614, -0.29973084, -0.29920782,
       -0.29872715, -0.29828896, -0.2978934 , -0.2975407 , -0.29723111,
       -0.29696494, -0.29674255, -0.29656434, -0.29643079, -0.2963424 ,
       -0.29629974, -0.29630344, -0.2963542 , -0.29645275, -0.29659992,
       -0.29679659, -0.29704371, -0.29734231, -0.29769351, -0.29809848,
       -0.29855852, -0.29907499, -0.29964935, -0.30028318, -0.30097817,
       -0.30173609, -0.30255889, -0.30344861, -0.30440745, -0.30543775,
       -0.30654202, -0.30772295, -0.3089834 ]
    return G0_j, G1_j, G2_j, G3_j
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

function get_g_tca()
    g_tca = [
       -0.45302768, -0.43408236, -0.41645932, -0.4000549 , -0.38477411,
       -0.37053003, -0.35724302, -0.3448402 , -0.3332548 , -0.3224257 ,
       -0.31229689, -0.30281702, -0.29393906, -0.28561986, -0.27781983,
       -0.27050264, -0.26363493, -0.25718605, -0.25112784, -0.24543439,
       -0.24008187, -0.23504836, -0.23031365, -0.22585914, -0.22166766,
       -0.21772341, -0.2140118 , -0.21051935, -0.20723364, -0.20414317,
       -0.20123736, -0.1985064 , -0.19594123, -0.19353349, -0.19127544,
       -0.18915993, -0.18718036, -0.18533062, -0.18360509, -0.18199855,
       -0.18050623, -0.17912369, -0.17784687, -0.17667203, -0.17559574,
       -0.17461486, -0.17372652, -0.17292809, -0.17221722, -0.17159174,
       -0.17104975, -0.1705895 , -0.17020949, -0.16990837, -0.169685  ,
       -0.16953837, -0.1694677 , -0.16947231, -0.16955173, -0.16970562,
       -0.16993379, -0.1702362 , -0.17061297, -0.17106436, -0.17159076,
       -0.17219273, -0.17287097, -0.17362631, -0.17445975, -0.17537242,
       -0.17636562, -0.1774408 , -0.17859956, -0.17984368, -0.1811751 ,
       -0.18259594, -0.18410849, -0.18571523, -0.18741884, -0.18922221,
       -0.19112841, -0.19314077, -0.19526282, -0.19749834, -0.19985139,
       -0.20232625, -0.20492752, -0.20766008, -0.21052912, -0.21354018,
       -0.21669912, -0.22001218, -0.223486  , -0.22712762, -0.23094452,
       -0.23494465, -0.23913645, -0.24352887, -0.24813143, -0.25295422,
       -0.25800799, -0.2633041 , -0.26885466, -0.27467252, -0.28077131,
       -0.28716553, -0.29387057, -0.30090278, -0.30827953, -0.31601927,
       -0.32414161, -0.33266736, -0.34161865, -0.35101898, -0.36089331,
       -0.37126814, -0.38217162, -0.39363362, -0.40568583, -0.41836187,
       -0.43169738, -0.44573013, -0.46050007, -0.47604948, -0.49242303,
       -0.50966783, -0.52783353, -0.54697232]
    -im * g_tca
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

    soi = ked.Hilbert.SetOfIndices([[1], [2]])
    ed = ked.EDCore(H_imp, soi)

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

    ρ_0 = full_hs_matrix(tofockbasis(density_matrix(expansion.P0), ed), ed)

    inchworm_matsubara!(expansion, grid, orders, orders_bare, N_samples)

    normalize!(expansion.P, β)
    ρ_wrm = full_hs_matrix(tofockbasis(density_matrix(expansion.P), ed), ed)

    ρ_exa = get_ρ_exact(ρ_wrm)
    ρ_nca = get_ρ_nca(ρ_wrm)
    ρ_oca = get_ρ_oca(ρ_wrm)
    ρ_tca = get_ρ_tca(ρ_wrm)

    diff_nca = maximum(abs.(ρ_wrm - ρ_nca))
    diff_oca = maximum(abs.(ρ_wrm - ρ_oca))
    diff_tca = maximum(abs.(ρ_wrm - ρ_tca))
    diff_exa = maximum(abs.(ρ_wrm - ρ_exa))

    ρ_000 = real(diag(ρ_0))
    ρ_exa = real(diag(ρ_exa))
    ρ_nca = real(diag(ρ_nca))
    ρ_oca = real(diag(ρ_oca))
    ρ_tca = real(diag(ρ_tca))
    ρ_wrm = real(diag(ρ_wrm))

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

    if true
    diff_g_nca = maximum(abs.(get_g_nca() - g[1].mat.data[1, 1, :]))
    diff_g_oca = maximum(abs.(get_g_oca() - g[1].mat.data[1, 1, :]))
    diff_g_tca = maximum(abs.(get_g_tca() - g[1].mat.data[1, 1, :]))

    if inch_print()

        @show diff_g_nca
        @show diff_g_oca
        @show diff_g_tca

    end
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
    for s in 1:4
        plt.plot(τ_ref, get_G_nca()[s], "k--", label="NCA $s ref", alpha=0.25)
    end
    for s in 1:4
        plt.plot(τ_ref, get_G_oca()[s], "k:", label="OCA $s ref", alpha=0.25)
    end
    for s in 1:4
        plt.plot(τ_ref, get_G_tca()[s], "k-.", label="TCA $s ref", alpha=0.25)
    end

    plt.ylabel(raw"$P_\Gamma(\tau)$")
    plt.xlabel(raw"$\tau$")
    plt.legend(loc="best")

    plt.subplot(subp...); subp[end] += 1;
    plt.plot(τ_ref, imag(get_g_nca()), label="NCA ref")
    plt.plot(τ_ref, imag(get_g_oca()), label="OCA ref")
    plt.plot(τ_ref, imag(get_g_tca()), label="TCA ref")
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

    @test diff_g_nca < 3e-3
    @test diff_g_nca < diff_g_oca

end

@testset "bethe_order2" begin

    ntau = 128
    orders = 0:2
    orders_gf = 0:1
    N_samples = 8 * 2^6
    μ_bethe = 0.25

    ρ, diffs_exa, diffs_nca, diffs_oca, diffs_tca, diff_g_nca, diff_g_oca =
        run_hubbard_dimer(ntau, orders, orders, orders_gf, N_samples, μ_bethe)

    @test diffs_oca < 2e-3
    @test diffs_oca < diffs_nca
    @test diffs_oca < diffs_tca
    @test diffs_oca < diffs_exa

    @test diff_g_oca < 1e-3
    @test diff_g_oca < diff_g_nca

end
