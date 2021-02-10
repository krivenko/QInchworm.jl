
module ppgf

import LinearAlgebra: Diagonal

import Keldysh: TimeGrid, TimeGF
import KeldyshED: EDCore, energies, partition_function

export AtomicPseudoParticleGF

function AtomicPseudoParticleGF(grid::TimeGrid, ed::EDCore, β::Real)

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

end # module ppgf
