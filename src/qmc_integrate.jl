module qmc_integrate

using MPI: MPI

import Keldysh; kd = Keldysh

using QInchworm.utility: get_ref
using QInchworm.utility: SobolSeqWith0, arbitrary_skip, next!
using QInchworm.utility: mpi_N_skip_and_N_samples_on_rank

#
# I use the notations introduced in https://arxiv.org/pdf/2002.12372.pdf.
#

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
    `seq`    Quasi-random sequence generator.
    `N`      The number of points taken from the quasi-random sequence.
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

"""
    MPI parallell Quasi Monte Carlo integration with warping.

    `f`      Integrand.
    `init`   Initial value of the integral.
    `p`      Positive model function p_n(u).
    `p_norm` Integral of p_n(u) over the u-domain.
    `trans`  Transformation from x ∈ [0,1]^d onto the u-domain.
    `seq`    Quasi-random sequence generator.
    `N`      The number of points taken from the quasi-random sequence.
"""
function qmc_integral_mpi(f, init = zero(typeof(f(trans(0))));
                      p, p_norm, trans, seq, N::Int)

    N_skip, N_samples_on_rank = mpi_N_skip_and_N_samples_on_rank(N)
    arbitrary_skip(seq, N_skip)

    # Eq. (5)
    res = init
    for i = 1:N_samples_on_rank
        x = next!(seq)
        u = trans(x)
        f_val = f(u)
        if isnothing(f_val) continue end
        res += f_val * (1.0 / p(u))
    end

    if isa(res, Dict)
        for (s_i, (s_f, mat)) in res
            mat[:] = MPI.Allreduce(mat, +, MPI.COMM_WORLD)
        end
    else
        # -- For supporting the qmc_integrate.jl tests
        res = MPI.Allreduce(res, +, MPI.COMM_WORLD)
    end

    (p_norm / N) * res
end

"""
    Quasi Monte Carlo integration with warping.

    This function takes a specified number of valid samples of the integrand.
    `nothing` returned by the integrand does not count towards this number.

    `f`         Integrand.
    `init`      Initial value of the integral.
    `p`         Positive model function p_n(u).
    `p_norm`    Integral of p_n(u) over the u-domain.
    `trans`     Transformation from x ∈ [0,1]^d onto the u-domain.
    `seq`       Quasi-random sequence generator.
    `N_samples` The number of taken samples.
"""
function qmc_integral_n_samples(f, init = zero(typeof(f(trans(0))));
                                p, p_norm, trans, seq, N_samples::Int)
    # Eq. (5)
    res = init
    i = 1
    N = 0
    while i <= N_samples
        N += 1
        x = next!(seq)
        u = trans(x)
        f_val = f(u)
        if isnothing(f_val) continue end
        res += f_val * (1.0 / p(u))
        i += 1
    end
    (p_norm / N) * res
end

const branch_direction = Dict(
    kd.forward_branch => 1.0,
    kd.backward_branch => -1.0,
    kd.imaginary_branch => -im
)

"""
    Detect the return type of a function applied to a vector of branch points.
"""
function contour_function_return_type(f)
    Base.return_types(f, (Vector{kd.BranchPoint},))[1]
end

raw"""
    Evaluate a d-dimensional contour-ordered integral of a function 'f',

    \int_{t_i}^{t_f} dt_1 \int_{t_i}^{t_1} dt_2 \ldots \int_{t_i}^{t_{d-1}} dt_d
        f(t_1, t_2, \ldots, t_d)

    using the Sobol sequence for quasi-random sampling.

    `f`    Integrand.
    `d`    Dimensionality of the integral.
    `c`    Time contour to integrate over.
    `t_i`  Starting time point on the contour.
    `t_f`  Final time point on the contour.
    `init` Initial value of the integral.
    `seq`  Quasi-random sequence generator.
    `τ`    Decay parameter of the exponential model function.
    `N`    The number of points taken from the quasi-random sequence.
"""
function qmc_time_ordered_integral(f,
                                   d::Int,
                                   c::kd.AbstractContour,
                                   t_i::kd.BranchPoint,
                                   t_f::kd.BranchPoint;
                                   init = zero(contour_function_return_type(f)),
                                   seq = SobolSeqWith0(d),
                                   τ::Real,
                                   N::Int)
    @assert kd.heaviside(c, t_f, t_i)

    # Model function, its norm and the x -> u transformation
    p_d = make_exp_model_function(c, t_f, τ, d)
    p_d_norm = exp_p_norm(τ, d)
    trans = make_exp_trans(c, t_f, τ)

    u_i = get_ref(c, t_i)

    N_samples::Int = 0

    (qmc_integral(init,
                 p = p_d,
                 p_norm = p_d_norm,
                 trans = trans,
                 seq = seq,
                 N = N) do refs
        refs[end] < u_i && return nothing # Discard irrelevant reference values
        N_samples += 1
        t_points = [c(r) for r in refs]
        coeff = prod(t -> branch_direction[t.domain], t_points)
        coeff * f(t_points)
    end, N_samples)
end

raw"""
    Evaluate a d-dimensional contour-ordered integral of a function 'f',

    \int_{t_i}^{t_f} dt_1 \int_{t_i}^{t_1} dt_2 \ldots \int_{t_i}^{t_{d-1}} dt_d
        f(t_1, t_2, \ldots, t_d)

    using the Sobol sequence for quasi-random sampling.

    This function evaluates the integrand a specified number of times.

    `f`         Integrand.
    `d`         Dimensionality of the integral.
    `c`         Time contour to integrate over.
    `t_i`       Starting time point on the contour.
    `t_f`       Final time point on the contour.
    `init`      Initial value of the integral.
    `seq`       Quasi-random sequence generator.
    `τ`         Decay parameter of the exponential model function.
    `N_samples` The number of taken samples.
"""
function qmc_time_ordered_integral_n_samples(
    f,
    d::Int,
    c::kd.AbstractContour,
    t_i::kd.BranchPoint,
    t_f::kd.BranchPoint;
    init = zero(contour_function_return_type(f)),
    seq = SobolSeqWith0(d),
    τ::Real,
    N_samples::Int)
    @assert kd.heaviside(c, t_f, t_i)

    # Model function, its norm and the x -> u transformation
    p_d = make_exp_model_function(c, t_f, τ, d)
    p_d_norm = exp_p_norm(τ, d)
    trans = make_exp_trans(c, t_f, τ)

    u_i = get_ref(c, t_i)

    N::Int = 0

    (qmc_integral_n_samples(init,
                            p = p_d,
                            p_norm = p_d_norm,
                            trans = trans,
                            seq = seq,
                            N_samples = N_samples) do refs
        N += 1
        refs[end] < u_i && return nothing # Discard irrelevant reference values
        t_points = [c(r) for r in refs]
        coeff = prod(t -> branch_direction[t.domain], t_points)
        coeff * f(t_points)
    end, N)
end

raw"""
    Evaluate a d-dimensional contour-ordered integral of a function 'f',

    \int_{t_i}^{t_f} dt_1 \int_{t_i}^{t_1} dt_2 \ldots \int_{t_i}^{t_{d-1}} dt_d
        f(t_1, t_2, \ldots, t_d)

    using the Sobol sequence for quasi-random sampling and the `Sort` transform.

    `f`    Integrand.
    `d`    Dimensionality of the integral.
    `c`    Time contour to integrate over.
    `t_i`  Starting time point on the contour.
    `t_f`  Final time point on the contour.
    `init` Initial value of the integral.
    `seq`  Quasi-random sequence generator.
    `N`    The number of points taken from the quasi-random sequence.
"""
function qmc_time_ordered_integral_sort(f,
                                        d::Int,
                                        c::kd.AbstractContour,
                                        t_i::kd.BranchPoint,
                                        t_f::kd.BranchPoint;
                                        init = zero(contour_function_return_type(f)),
                                        seq = SobolSeqWith0(d),
                                        N::Int)
    @assert kd.heaviside(c, t_f, t_i)

    # x -> u transformation
    u_i = get_ref(c, t_i)
    u_f = get_ref(c, t_f)
    ref_diff = u_f - u_i
    trans = x -> u_i .+ sort(x, rev=true) * ref_diff

    qmc_integral(init,
                 p = u -> 1.0,
                 p_norm = (ref_diff ^ d) / factorial(d),
                 trans = trans,
                 seq = seq,
                 N = N) do refs
        t_points = [c(r) for r in refs]
        coeff = prod(t -> branch_direction[t.domain], t_points)
        coeff * f(t_points)
    end
end

raw"""
    Evaluate a d-dimensional contour-ordered integral of a function 'f',

    \int_{t_i}^{t_f} dt_1 \int_{t_i}^{t_1} dt_2 \ldots \int_{t_i}^{t_{d-1}} dt_d
        f(t_1, t_2, \ldots, t_d)

    using the Sobol sequence for quasi-random sampling and the `Root` transform.

    `f`    Integrand.
    `d`    Dimensionality of the integral.
    `c`    Time contour to integrate over.
    `t_i`  Starting time point on the contour.
    `t_f`  Final time point on the contour.
    `init` Initial value of the integral.
    `seq`  Quasi-random sequence generator.
    `N`    The number of points taken from the quasi-random sequence.
"""
function qmc_time_ordered_integral_root(f,
                                        d::Int,
                                        c::kd.AbstractContour,
                                        t_i::kd.BranchPoint,
                                        t_f::kd.BranchPoint;
                                        init = zero(contour_function_return_type(f)),
                                        seq = SobolSeqWith0(d),
                                        N::Int)
    @assert kd.heaviside(c, t_f, t_i)

    # x -> u transformation
    u_i = get_ref(c, t_i)
    u_f = get_ref(c, t_f)
    ref_diff = u_f - u_i

    trans = x -> begin
        u = Vector{Float64}(undef, d)
        u[1] = x[1] ^ (1.0 / d)
        for s = 2:d
            u[s] = u[s - 1] * (x[s] ^ (1.0 / (d - s + 1)))
        end
        return u_i .+ u * ref_diff
    end

    qmc_integral_mpi(init,
                 p = u -> 1.0,
                 p_norm = (ref_diff ^ d) / factorial(d),
                 trans = trans,
                 seq = seq,
                 N = N) do refs
        t_points = [c(r) for r in refs]
        coeff = prod(t -> branch_direction[t.domain], t_points)
        coeff * f(t_points)
    end
end

raw"""
    Evaluate an inchworm-type contour-ordered integral of a function 'f',

    \int_{t_w}^{t_f} dt_1
    \int_{t_w}^{t_1} dt_2 \ldots
    \int_{t_w}^{t_{d_{after}-1}} dt_{d_{after}}
    \int_{t_i}^{t_w} dt_{d_{after}+1} \ldots
    \int_{t_i}^{t_{d-2}} dt_{d-1}
    \int_{t_i}^{t_{d-1}} dt_d
        f(t_1, t_2, \ldots, t_d)

    using the Sobol sequence for quasi-random sampling and a two-piece `Root`
    transform.
    The total dimension of the domain `d` is a sum of the amount of integration
    variables in the 'before' (`d_{before}`) and the after (`d_{after}`) components.

Parameters
----------

f : Integrand.
d_before : Dimensionality of the before-t_w component of the integration domain.
d_after : Dimensionality of the after-t_w component of the integration domain.
c : Time contour to integrate over.
t_i : Starting time point on the contour.
t_w : 'Worm' time point on the contour separating 'before' and 'after' parts.
t_f : Final time point on the contour.
init : Initial value of the integral.
seq : Quasi-random sequence generator.
N : The number of samples.

Returns
-------

Value of the integral.
"""
function qmc_inchworm_integral_root(f,
                                    d_before::Int,
                                    d_after::Int,
                                    c::kd.AbstractContour,
                                    t_i::kd.BranchPoint,
                                    t_w::kd.BranchPoint,
                                    t_f::kd.BranchPoint;
                                    init = zero(contour_function_return_type(f)),
                                    seq = SobolSeqWith0(d_before + d_after),
                                    N::Int)
    @assert kd.heaviside(c, t_w, t_i)
    @assert kd.heaviside(c, t_f, t_w)
    @assert d_before >= 0
    @assert d_after >= 1

    u_i = get_ref(c, t_i)
    u_w = get_ref(c, t_w)
    u_f = get_ref(c, t_f)

    ref_diff_wi = u_w - u_i
    ref_diff_fw = u_f - u_w

    d = d_before + d_after

    # QUESTION: Currently we apply the two-piece transformation to the same
    # quasi-random sequence of points in d dimensions. Would it make more sense
    # to use two independent sequences with dimensions d_after and d_before?

    u = Vector{Float64}(undef, d)
    # x -> u transformation
    trans = x -> begin
        # Points in the bare region
        u[1] = x[1] ^ (1.0 / d_after)
        for s = 2:d_after
            u[s] = u[s - 1] * (x[s] ^ (1.0 / (d_after - s + 1)))
        end
        u[1:d_after] *= ref_diff_fw
        u[1:d_after] .+= u_w

        d_before == 0 && return u

        # Points in the bold region
        u[d_after + 1] = x[d_after + 1] ^ (1.0 / d_before)
        for s = (d_after + 2):d
            u[s] = u[s - 1] * (x[s] ^ (1.0 / (d - s + 1)))
        end
        u[d_after+1:d] *= ref_diff_wi
        u[d_after+1:d] .+= u_i

        return u
    end

    p_norm = (ref_diff_fw ^ d_after) / factorial(d_after) *
             (ref_diff_wi ^ d_before) / factorial(d_before)

    qmc_integral_mpi(init,
                 p = u -> 1.0,
                 p_norm = p_norm,
                 trans = trans,
                 seq = seq,
                 N = N) do refs
        t_points = [c(r) for r in refs]
        coeff = prod(t -> branch_direction[t.domain], t_points)
        coeff * f(t_points)
    end
end

end # module qmc_integrate
