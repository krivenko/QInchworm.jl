module ppgf

import LinearAlgebra: Diagonal, tr, I, diagm

import Keldysh; kd = Keldysh;
import KeldyshED; ked = KeldyshED;

import KeldyshED: EDCore, energies, partition_function
import KeldyshED: c_connection, cdag_connection
import KeldyshED: c_matrix, cdag_matrix

import QInchworm.spline_gf: SplineInterpolatedGF
using  QInchworm.utility: inch_print

export FullTimePPGF, ImaginaryTimePPGF
export atomic_ppgf
export operator_product

"""
Get matrix representation of operator expression in each sector

NB! Requires that the operator expression does not mix symmetry sectors
"""
function operator_matrix_representation(
    op_expr::ked.OperatorExpr{S}, ed::ked.EDCore) where {S <: Number}

    op = ked.Operator{ked.FullHilbertSpace, S}(op_expr, ed.full_hs.soi)

    op_sector_matrices = Matrix{S}[]
    for (sidx, subspace) in enumerate(ed.subspaces)
        op_matrix = Matrix{S}(undef, length(subspace), length(subspace))
        i_state = ked.StateVector{ked.HilbertSubspace, S}(subspace)
        for i in 1:length(subspace)
            i_state[i] = one(S)
            f_state = op * i_state
            op_matrix[:, i] = f_state.amplitudes
            i_state[i] = zero(S)
        end
        push!(op_sector_matrices, op_matrix)
    end
    op_sector_matrices
end

function total_density_operator(ed::ked.EDCore)
    sum([ ked.Operators.n(label...) for (label, i) in ed.full_hs.soi ])
end

# N.B. We cannot use FullTimeGF instead of GenericTimeGF here,
# because FullTimeGF's data storage scheme relies on the symmetry
# properties the pseudo-particle GF's do not possess.
const FullTimePPGFSector = kd.GenericTimeGF{ComplexF64, false, kd.FullTimeGrid}
const FullTimePPGF = Vector{FullTimePPGFSector}
const ImaginaryTimePPGFSector = kd.ImaginaryTimeGF{ComplexF64, false}
const ImaginaryTimePPGF = Vector{ImaginaryTimePPGFSector}

const AllImaginaryTimeGF = Union{
    kd.ImaginaryTimeGF{ComplexF64, false},
    SplineInterpolatedGF{kd.ImaginaryTimeGF{ComplexF64, false}, ComplexF64, false}
}

"""
Compute atomic pseudo-particle Green's function on the time grid
for a time-independent problem defined by the EDCore instance.
"""
function atomic_ppgf(grid::kd.FullTimeGrid, ed::EDCore)::FullTimePPGF
    G = [kd.GenericTimeGF(grid, length(s)) for s in ed.subspaces]
    atomic_ppgf!(G, ed)
    G
end

function atomic_ppgf(grid::kd.ImaginaryTimeGrid, ed::EDCore)::ImaginaryTimePPGF
    G = [kd.ImaginaryTimeGF(grid, length(s)) for s in ed.subspaces]
    atomic_ppgf!(G, ed)
    G
end

function atomic_ppgf!(G::Vector, ed::EDCore)
    @assert length(G) == length(ed.subspaces)

    β = G[1].grid.contour.β
    Z = partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[G0(β)]=Tr[ρ]=1)

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for (G_s, s, E, n) in zip(G, ed.subspaces, energies(ed), N)
        ξ = (-1)^n[1,1] # Statistics sign
        grid = G_s.grid
        z_β = grid[kd.imaginary_branch][end]
        Threads.@threads for z1 in grid
	        for z2 in grid[1:z1.cidx]
                Δz = z1.bpoint.val - z2.bpoint.val
                if z1.bpoint.domain == kd.forward_branch &&
                    z2.bpoint.domain != kd.forward_branch
                    Δz += -im*β
                end
                sign = ξ^(z1.cidx > z_β.cidx && z_β.cidx >= z2.cidx)
                G_s[z1, z2] = -im * sign * Diagonal(exp.(-im * Δz * (E .+ λ)))
	        end
        end
    end
end


"""
    operator_product(...)

Evaluate a product of vertices at different contour times `z_i` with
the pseudo-particle Green's function sandwitched in between.

`vertices` is a contour-time ordered list of triples `(z_i, c_i, o_i)` were:
  `z_i` is a contour time,
  `c_i` is +1/-1 for creation/annihilation operator respectively, and
  `o_i` is a spin-orbital index
"""
function operator_product(ed::EDCore, G, s_i::Integer, z_i, z_f, vertices)

    length(vertices) == 0 && return G[s_i][z_f, z_i], s_i

    s_a = s_i
    (z_a, c_a, o_a) = vertices[1]

    prod0 = im * G[s_a][z_a, z_i]
    prod = prod0

    for (vidx, (z_a, c_a, o_a)) in enumerate(vertices)

        connection = c_a > 0 ? cdag_connection : c_connection
        matrix = c_a > 0 ? cdag_matrix : c_matrix

        s_b = connection(ed, o_a, s_a)
        s_b === nothing && return 0 * prod0, -1

        m_ba = matrix(ed, o_a, s_a)

        if vidx < length(vertices)
            z_b = vertices[vidx + 1][1]
        else
            z_b = z_f
        end

        prod = im * G[s_b][z_b, z_a] * m_ba * prod

        s_a = s_b
    end

    prod, s_a
end

"""
Compute the first order pseudo-particle diagram contribution to
the single-particle Green's function g_{o1, o2}(z, z')
"""
function first_order_spgf(ppgf::FullTimePPGF, ed::ked.EDCore, o1, o2)::kd.FullTimeGF
    @assert length(ppgf) == length(ed.subspaces)
    g = kd.FullTimeGF(ppgf[1].grid, 1, kd.fermionic, true)
    first_order_spgf!(g, ppgf, ed, o1, o2)
    g
end

function first_order_spgf(ppgf::ImaginaryTimePPGF, ed::ked.EDCore, o1, o2)::kd.ImaginaryTimeGF
    @assert length(ppgf) == length(ed.subspaces)
    g = kd.ImaginaryTimeGF(ppgf[1].grid, 1, kd.fermionic, true)
    first_order_spgf!(g, ppgf, ed, o1, o2)
    g
end

function first_order_spgf!(g, ppgf, ed::ked.EDCore, o1, o2)
    @assert length(ppgf) == length(ed.subspaces)

    grid = g.grid

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for z1 in grid, z2 in grid

        # Creation/annihilator operator commutation sign
        sign = (-1)^(z1.cidx < z2.cidx)

        # Operator verticies
        v1 = (z1, -1, o1)
        v2 = (z2, +1, o2)

        # (twisted contour) time ordered operator verticies
        v1, v2 = sort([v1, v2], by = x -> x[1].cidx, rev=true)

        # -- Determine start and end time on twisted contour
        if z1.bpoint.domain == kd.imaginary_branch && z2.bpoint.domain == kd.imaginary_branch
            # Equilibrium start at τ = 0 and end at τ = β
            real_time = false
            tau_grid = grid[kd.imaginary_branch]
            z_i, z_f = tau_grid[1], tau_grid[end]
        else
            # Atleast one time is in real-time
            real_time = true

            z_max = sort([z1, z2],
                by = x -> real(x.bpoint.val) - (x.bpoint.domain == kd.imaginary_branch))[end]

            if z_max.bpoint.domain == kd.forward_branch
                z_f = z_max
                z_i = grid[1 + grid[end].cidx - z_max.cidx]
            else
                z_i = z_max
                z_f = grid[1 + grid[end].cidx - z_max.cidx]
            end
        end

        g[z1, z2] = .0
        for (sidx, s) in enumerate(ed.subspaces)
            ξ = (-1)^N[sidx][1, 1]
            prod, sidx_f = operator_product(ed, ppgf, sidx, z_i, z_f, [v2, v1])
            if sidx == sidx_f
                g[z1, z2] += -im * sign * ξ^real_time * tr(prod)
            end
        end
    end
end

function check_ppgf_real_time_symmetries(G::FullTimePPGF, ed)
    @assert length(G) == length(ed.subspaces)
    grid = G[1].grid

    grid_bwd = grid[kd.backward_branch]
    zb_i, zb_f = grid_bwd[1], grid_bwd[end]

    grid_fwd = grid[kd.forward_branch]
    zf_i, zf_f = grid_fwd[1], grid_fwd[end]

    # Symmetry between G_{--} and G_{++}

    for zb_1 in grid_bwd
        for zb_2 in grid_bwd[1:zb_1.cidx]
            @assert zb_1.cidx >= zb_2.cidx

            zf_1 = grid[zf_f.cidx - zb_1.cidx + 1]
            zf_2 = grid[zf_f.cidx - zb_2.cidx + 1]

            @assert zb_1.bpoint.val ≈ zf_1.bpoint.val
            @assert zb_2.bpoint.val ≈ zf_2.bpoint.val

            @assert zf_2.cidx >= zf_1.cidx

            for g_s in G
                @assert g_s[zb_1, zb_2] ≈ -conj(g_s[zf_2, zf_1])
            end
        end
    end

    # Symmetry along anti-diagonal of G_{+-}

    for zf_1 in grid_fwd
        for zb_1 in grid_bwd[1:zf_f.cidx - zf_1.cidx + 1]

            zf_2 = grid[zf_f.cidx - (zb_1.cidx - zb_i.cidx)]
            zb_2 = grid[zb_f.cidx - (zf_1.cidx - zf_i.cidx)]

            @assert zf_1.bpoint.val ≈ zb_2.bpoint.val
            @assert zb_1.bpoint.val ≈ zf_2.bpoint.val

            for g_s in G
                @assert g_s[zf_1, zb_1] ≈ -conj(g_s[zf_2, zb_2])
            end
        end
    end

    # Symmetry between G_{M-} and G_{+M}

    z_0 = grid[kd.imaginary_branch][1]
    z_β = grid[kd.imaginary_branch][end]
    β = im * z_β.bpoint.val

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    for zf in grid_bwd

        zb = grid[zf_f.cidx - zf.cidx + 1]
        @assert zf.bpoint.val ≈ zb.bpoint.val

        for τ in grid[kd.imaginary_branch]
            τ_β = grid[z_β.cidx - (τ.cidx - z_0.cidx)]
            @assert τ_β.bpoint.val ≈ -im*β - τ.bpoint.val

            for (sidx, g_s) in enumerate(G)
                ξ = (-1)^N[sidx][1, 1]
                @assert g_s[τ, zf] ≈ -ξ * conj(g_s[zb, τ_β])
            end
        end
    end
    return true
end

function set_ppgf_initial_conditions!(G::Union{FullTimePPGF, ImaginaryTimePPGF})
    for g in G
        g = zero(g)
        for z in g.grid
            g[z, z] += -im * I
        end
    end
end

function ppgf_real_time_initial_conditions!(G::FullTimePPGF, ed::ked.EDCore)
    @assert length(G) == length(ed.subspaces)

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)

    grid = G[1].grid
    zb0 = grid[kd.backward_branch][end]
    zf0 = grid[kd.forward_branch][1]
    τ_0 = grid[kd.imaginary_branch][1]
    τ_β = grid[kd.imaginary_branch][end]

    for (G_s, n) in zip(G, N)
        for τ in grid[kd.imaginary_branch]
            set_ppgf_symmetric!(G_s, n, τ, zb0, G_s[τ, τ_0])
        end
        G_s[zf0, zb0] = (-1)^n[1,1] * G_s[τ_β, τ_0]
    end
end

""" Set real-time ppgf symmetry connected time pairs

NB! times has to be in the inching region with z2 ∈ backward_branch. """
function set_ppgf_symmetric!(G_s::FullTimePPGF, n, z1, z2, val)
     grid = G_s.grid

    grid_bwd = grid[kd.backward_branch]
    zb_i, zb_f = grid_bwd[1], grid_bwd[end]

    grid_fwd = grid[kd.forward_branch]
    zf_i, zf_f = grid_fwd[1], grid_fwd[end]

    z_0 = grid[kd.imaginary_branch][1]
    z_β = grid[kd.imaginary_branch][end]

    η = 1

    if z1.bpoint.domain == kd.backward_branch &&
       z2.bpoint.domain == kd.backward_branch
        z3 = grid[zf_f.cidx - z2.cidx + 1]
        z4 = grid[zf_f.cidx - z1.cidx + 1]
    elseif z1.bpoint.domain == kd.imaginary_branch &&
           z2.bpoint.domain == kd.backward_branch
        z3 = grid[zf_f.cidx - z2.cidx + 1]
        z4 = grid[z_β.cidx - (z1.cidx - z_0.cidx)]
        η = -1
    elseif z1.bpoint.domain == kd.forward_branch &&
           z2.bpoint.domain == kd.backward_branch
        z3 = grid[zf_f.cidx - (z2.cidx - zb_i.cidx)]
        z4 = grid[zb_f.cidx - (z1.cidx - zf_i.cidx)]
    else
        @assert false
    end

    ξ = η^n[1, 1]
    G_s[z3, z4] = -ξ * conj(val)
    G_s[z1, z2] = val
end

function partition_function(G::Vector{<:kd.AbstractTimeGF})
    sum(G, init = 0im) do G_s
        g_s = G_s[kd.imaginary_branch, kd.imaginary_branch]
        g_s = vcat(g_s[:, 1]...)
        im * tr(g_s[end])
    end
end

"""Set all time translation invariant values of the Matsubara branch"""
function set_matsubara!(g::kd.GenericTimeGF{T, scalar, kd.FullTimeGrid} where {T, scalar}, τ, value)
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

function set_matsubara!(g::AllImaginaryTimeGF, τ, value)
    tau_grid = g.grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    g[τ, τ_0] = value
end

function normalize!(G::Vector{<:kd.AbstractTimeGF}, β)
    Z = partition_function(G)
    λ = log(Z) / β
    for g in G
        normalize!(g, λ)
    end
end

function normalize!(g::kd.AbstractTimeGF, λ)
    tau_grid = g.grid[kd.imaginary_branch]
    τ_0 = tau_grid[1]
    for τ in tau_grid
        val = g[τ, τ_0] .* exp(-1im * τ.bpoint.val * λ)
        set_matsubara!(g, τ, val)
    end
end

function initial_ppgf_derivative(ed::ked.EDCore, β::Float64)
    Z = sum([ sum(exp.(-β * eig.eigenvalues)) for eig in ed.eigensystems ])
    if inch_print(); @show Z; end
    λ = log(Z) / β
    if inch_print(); @show λ; end

    dP = []
    for eig in ed.eigensystems
        if inch_print(); @show eig.eigenvalues; end
        dP_s = -im * diagm(-eig.eigenvalues .- λ)
        push!(dP, dP_s)
    end
    return dP
end

end # module ppgf
