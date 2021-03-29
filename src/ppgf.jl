
module ppgf

import LinearAlgebra: Diagonal, tr

import Keldysh: TimeGrid, TimeGF
import Keldysh; kd = Keldysh;

import KeldyshED: EDCore, energies, partition_function
import KeldyshED: c_connection, cdag_connection
import KeldyshED: c_matrix, cdag_matrix

import KeldyshED; ked = KeldyshED;

export atomic_ppgf
export operator_product


"""
Get matrix representation of operator expression in each sector

NB! Requires that the operator expression does not mix symmetry sectors
"""
function operator_matrix_representation(
    op_expr::ked.OperatorExpr{S}, ed::ked.EDCore) where {S <: Number}

    op = ked.Operator{ked.FullHilbertSpace, S}(op_expr, ed.full_hs.soi)
    
    op_sector_matrices = Vector{Matrix{S}}()    
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
    N = sum([ ked.Operators.n(label...) for (label, i) in ed.full_hs.soi ])
end


"""
Compute atomic pseudo-particle Green's function on the time grid
for a time-independent problem defined by the EDCore instance.
"""
function atomic_ppgf(grid::TimeGrid, ed::EDCore, β::Real)

    G = Vector{TimeGF}()

    N_op = total_density_operator(ed)
    N = operator_matrix_representation(N_op, ed)
    z_β = grid[kd.imaginary_branch][end]
    
    Z = partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[G0(β)]=Tr[ρ]=1)
    
    for (s, E, n) in zip(ed.subspaces, energies(ed), N)
        G_s = TimeGF(grid, length(s))
        ξ = (-1)^n[1,1] # Statistics sign
        for z1 in grid, z2 in grid[1:z1.idx]
            Δz = z1.val.val - z2.val.val
            if z1.val.domain == kd.forward_branch && 
               z2.val.domain != kd.forward_branch                
                Δz += -im*β
            end            
            sign = ξ^(z1.idx > z_β.idx && z_β.idx >= z2.idx)
            G_s[z1, z2] = -im * sign * Diagonal(exp.(-im * Δz * (E .+ λ)))
        end
        push!(G, G_s)
    end    
    return G
end


"""
    operator_product(...)

Evaluate a product of vertices at different contour times `z_i` with 
the pseud-particle Green's function sandwitched in between.

`vertices` is a contour-time ordered list of triples `(z_i, c_i, o_i)` were:
  `z_i` is a contour time,
  `c_i` is +1/-1 for creation/annihilation operator respectively, and 
  `o_i` is a spin-orbital index
"""
function operator_product(ed::EDCore, G, s_i::Integer, z_i, z_f, vertices)

    length(vertices) == 0 && return G[s_i][z_f, z_i]

    s_a = s_i
    (z_a, c_a, o_a) = vertices[1]

    prod0 = im * G[s_a][z_a, z_i]
    prod = prod0
    
    for (vidx, (z_a, c_a, o_a)) in enumerate(vertices)

        connection = c_a > 0 ? cdag_connection : c_connection
        matrix = c_a > 0 ? cdag_matrix : c_matrix

        s_b = connection(ed, o_a, s_a)
        s_b == nothing && return 0 * prod0

        m_ba = matrix(ed, o_a, s_a)

        if vidx < length(vertices)
            z_b = vertices[vidx + 1][1]
        else
            z_b = z_f
        end

        prod = im * G[s_b][z_b, z_a] * m_ba * prod

        s_a = s_b
    end

    s_a != s_i && return 0 * prod0

    prod
end


"""
Compute the first order pseudi-particle diagram contribution to 
the single-particle Green's function g_{o1, o2}(z, z')
"""
function first_order_spgf(ppgf::Vector{S}, ed::ked.EDCore, o1, o2) where {S <: kd.TimeGF}

    grid = ppgf[1].grid
    g = kd.TimeGF(grid)
    
    for z1 in grid, z2 in grid

        sign = 1
    
        # Operator verticies
        v1 = (z1, +1, o2)
        v2 = (z2, -1, o1)
        
        # (twisted contour) time ordered operator verticies
        v1, v2 = sort([v1, v2], by = x -> x[1].idx)

        # -- Determine start and end time on twisted contour
        if z1.val.domain == kd.imaginary_branch && z2.val.domain == kd.imaginary_branch
            # Equilibrium start at \tau = 0 and end at \tau = \beta
            tau_grid = grid[kd.imaginary_branch]
            z_i, z_f = tau_grid[1], tau_grid[end]
        
            # Creation/annihilator operator commutation sign
            # why only on imaginary time branch?
            sign *= -(-1)^(z1.idx <= z2.idx)
        else
            # Atleast one time is in real-time            
            z_max = sort([z1, z2], 
                by = x -> real(x.val.val) - (x.val.domain == kd.imaginary_branch))[end]
            
            if z_max.val.domain == kd.forward_branch
                z_f = z_max
                z_i = grid[1 + grid[end].idx - z_max.idx]
            else
                z_i = z_max
                z_f = grid[1 + grid[end].idx - z_max.idx]
            end
        end
    
        for (sidx, s) in enumerate(ed.subspaces)
            prod = operator_product(ed, ppgf, sidx, z_i, z_f, [v1, v2])
            g[z2, z1] += -im * sign * tr(prod)
        end
    end
    return g
end


end # module ppgf
