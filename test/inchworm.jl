using Test

import Keldysh; kd = Keldysh
import KeldyshED; ked = KeldyshED; op = KeldyshED.Operators;

import QInchworm.configuration: Expansion, InteractionPair
import QInchworm.inchworm: inchworm_step

# -- Single state pseudo particle expansion

β = 10.
μ = +0.1 # Chemical potential
ϵ = +0.1 # Bath energy level
V = -0.1 # Hybridization
nt = 10
ntau = 30
tmax = 1.

H = - μ * op.n("0")
soi = KeldyshED.Hilbert.SetOfIndices([["0"]])
ed = KeldyshED.EDCore(H, soi)
ρ = KeldyshED.density_matrix(ed, β)

contour = kd.twist(kd.FullContour(tmax=tmax, β=β));
grid = kd.FullTimeGrid(contour, nt, ntau);

# -- Hybridization propagator

Δ = kd.FullTimeGF(
    (t1, t2) -> -1.0im * V^2 *
        (kd.heaviside(t1.bpoint, t2.bpoint) - kd.fermi(ϵ, contour.β)) *
        exp(-1.0im * (t1.bpoint.val - t2.bpoint.val) * ϵ),
    grid, 1, kd.fermionic, true)

# -- Pseudo Particle Strong Coupling Expansion

ip_fwd = InteractionPair(op.c_dag("0"), op.c("0"), Δ)
ip_bwd = InteractionPair(op.c("0"), op.c_dag("0"), Δ)
expansion = Expansion(ed, grid, [ip_fwd, ip_bwd])

@testset "inchworm_step" begin
    # Fixing the initial, final and worm-time
    i_idx = 1
    f_idx = 8
    w_idx = f_idx - 1

    τ_grid = grid[kd.imaginary_branch]
    τ_i = τ_grid[i_idx].bpoint
    τ_w = τ_grid[w_idx].bpoint
    τ_f = τ_grid[f_idx].bpoint

    orders = 0:3
    N_samples = fill(1000, length(orders))

    value = inchworm_step(expansion, contour, τ_w, τ_w, τ_f, orders, N_samples)
    println("value w/o bold part = $value")
    value = inchworm_step(expansion, contour, τ_i, τ_w, τ_f, orders, N_samples)
    println("value with bold part = $value")
end
