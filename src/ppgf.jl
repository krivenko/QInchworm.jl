
module ppgf

import LinearAlgebra: Diagonal

import Keldysh: TimeGrid, TimeGF

import KeldyshED: EDCore, energies, partition_function
import KeldyshED: c_connection, cdag_connection
import KeldyshED: c_matrix, cdag_matrix

export atomic_ppgf
export operator_product

function atomic_ppgf(grid::TimeGrid, ed::EDCore, β::Real)

    G = Vector{TimeGF}()
    
    Z = partition_function(ed, β)
    λ = log(Z) / β # Pseudo-particle chemical potential (enforcing Tr[G0(β)]=Tr[ρ]=1)
    
    for (s, E) in zip(ed.subspaces, energies(ed))
        G_s = TimeGF(grid, length(s))
        for t1 in grid, t2 in grid
            Δt = t1.val.val - t2.val.val
            G_s[t1, t2] = Diagonal(exp.(-1im * Δt * (E .+ λ)))
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

    prod0 = G[s_a][z_a, z_i]
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

        prod = G[s_b][z_b, z_a] * m_ba * prod

        s_a = s_b
    end

    s_a != s && return 0 * prod0

    prod
end;

end # module ppgf
