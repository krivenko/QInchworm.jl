module qmc_integrate

import Sobol: SobolSeq, next!
import Keldysh; kd = Keldysh

#
# I use the notations introduced in https://arxiv.org/pdf/2002.12372.pdf.
#

"""
    Inverse of get_point(c::AbstractContour, ref)

    TODO: Ask Joseph to promote it to Keldysh.jl?
"""
function get_ref(c::kd.AbstractContour, t::kd.BranchPoint)
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

"""
    Make the model function p_d(u) out of h_i(v).
"""
function make_model_function(c::kd.AbstractContour,
                             t_f::kd.BranchPoint,
                             h::Vector)
    u_f = get_ref(c, t_f)
    # Eq. (6)
    u::Vector{Float64} -> begin
        # Transformation u -> v
        v = similar(u)
        v[1] = u_f - u[1]
        v[2:end] = -diff(u)
        # Product of h(v_i)
        reduce(*, [hi(vi) for (hi, vi) in zip(h, v)])
    end
end

# TODO: Bind make_exp_model_function(), exp_p_norm() and make_exp_trans()
# together.

"""
    Make p_d(u) from the exponential h(v).
"""
function make_exp_model_function(c::kd.AbstractContour,
                                 t_f::kd.BranchPoint,
                                 τ::Real,
                                 d::Int)
    # FIXME
    #make_model_function(c, t_f, repeat([v -> exp(-v/τ)], d))

    u_f = get_ref(c, t_f)
    # Eq. (6)
    u::Vector{Float64} -> begin
        # Transformation u -> v

        #v = similar(u)
        #v[1] = u_f - u[1]
        #v[2:end] = -diff(u)
        #exp(-sum(v) / τ)

        # Simplification: \sum_i exp(-v_i / τ) = exp(-(u_f - u[end]) / τ)
        exp(-(u_f - u[end]) / τ)
    end
end

raw"""
    Make the corresponding transformation x -> u

    x ∈ [0,1]^d and components of u are reference values of time points measured
    as distances along the contour (c.f. get_point(c::AbstractContour, ref)).
    These components satisfy

        u_f > u_1 > u_2 > \ldots > u_{d-1} > u_d > -\infty.
"""
function make_exp_trans(c::kd.AbstractContour, t_f::kd.BranchPoint, τ::Real)
    u_f = get_ref(c, t_f)
    x -> begin
        u = Vector{Float64}(undef, length(x))
        u[1] = u_f + τ*log(1 - x[1])
        for i = 2:length(x)
            u[i] = u[i-1] + τ*log(1 - x[i])
        end
        u
    end
end

"""
    Normalization of the model function p_d(u)
"""
exp_p_norm(τ::Real, d::Int) = τ^d

"""
    Quasi Monte Carlo integration with warping.

    `f`      Integrand.
    `init`   Initial value of the integral.
    `p`      Positive model function p_n(u).
    `p_norm` Integral of p_n(u) over the u-domain.
    `trans`  Transformation from x ∈ [0,1]^d onto the u-domain.
    `seq`    Pseudo-random sequence generator.
    `N`      The number of taken samples.
"""
function qmc_integral(f, init = zero(typeof(f(trans(0))));
                      p, p_norm, trans, seq, N::Int)
    # Eq. (5)
    res = init
    for i = 1:N
        x = next!(seq)
        u = trans(x)
        f_val = f(u)
        if isnothing(f_val) continue end
        res += f_val * (1.0 / p(u))
    end
    (p_norm / N) * res
end

const branch_direction = Dict(
    kd.forward_branch => 1.0,
    kd.backward_branch => -1.0,
    kd.imaginary_branch => -im
)

raw"""
    Evaluate a d-dimensional contour-ordered integral of a function 'f',

    \int_{t_i}^{t_f} dt_1 \int_{t_i}^{t_1} dt_2 \ldots \int_{t_i}^{t_{d-1}} dt_d
        f(t_1, t_2, \ldots, t_d)

    using the Sobol sequence for pseudo-random sampling.

    `f`    Integrand.
    `d`    Dimensionality of the integral.
    `c`    Time contour to integrate over.
    `t_i`  Starting time point on the contour.
    `t_f`  Final time point on the contour.
    `init` Initial value of the integral.
    `seq`  Quasi-random sequence generator.
    `τ`    Decay parameter of the exponential model function.
    `N`    The number of taken samples.
"""
function qmc_time_ordered_integral(f,
                                   d::Int,
                                   c::kd.AbstractContour,
                                   t_i::kd.BranchPoint,
                                   t_f::kd.BranchPoint;
                                   init = zero(typeof(f(repeat([t_i], d)))),
                                   seq = SobolSeq(d),
                                   τ::Real,
                                   N::Int)
    @assert kd.heaviside(c, t_f, t_i)

    # Model function, its norm and the x -> u transformation
    p_d = make_exp_model_function(c, t_f, τ, d)
    p_d_norm = exp_p_norm(τ, d)
    trans = make_exp_trans(c, t_f, τ)

    u_i = get_ref(c, t_i)

    qmc_integral(init,
                 p = p_d,
                 p_norm = p_d_norm,
                 trans = trans,
                 seq = seq,
                 N = N) do refs
        refs[end] < u_i && return nothing # Discard irrelevant reference values
        t_points = [c(r) for r in refs]
        coeff = prod(t -> branch_direction[t.domain], t_points)
        coeff * f(t_points)
    end
end

end # module qmc_integrate
