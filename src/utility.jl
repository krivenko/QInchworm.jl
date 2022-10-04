module utility

import Keldysh

"""
    Inverse of get_point(c::AbstractContour, ref)

    TODO: Ask Joseph to promote it to Keldysh.jl?
"""
function get_ref(c::Keldysh.AbstractContour, t::Keldysh.BranchPoint)
    ref = 0
    for b in c.branches
        lb = length(b)
        if t.domain == b.domain
            return ref + (t.ref * lb)
        else
            ref += lb
        end
    end
    @assert false
end

end