# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
#
# QInchworm.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# QInchworm.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Igor Krivenko, Hugo U. R. Strand

"""
Quasi Monte Carlo integration routines.

Concepts and notation used here are introduced in

```
Quantum Quasi-Monte Carlo Technique for Many-Body Perturbative Expansions
M. Maček, P. T. Dumitrescu, C. Bertrand, B.Triggs, O. Parcollet, and X. Waintal
Phys. Rev. Lett. 125, 047702 (2020)
```

Integration domain transformations `Sort` and `Root` are defined in
```
Transforming low-discrepancy sequences from a cube to a simplex
T. Pillards and R. Cools
J. Comput. Appl. Math. 174, 29 (2005)
```
"""
module qmc_integrate

using DocStringExtensions

using MPI: MPI

using Keldysh; kd = Keldysh

using QInchworm.utility: SobolSeqWith0, next!

"""
    $(TYPEDSIGNATURES)

Make the model function ``p_d(\\mathbf{u}) = \\prod_{i=1}^d h_i(u_{i-1} - u_i)``
out of a list of functions ``h_i(v)``.

``u_0`` is the distance measured along the time contour `c` to the point `t_f`.
"""
function make_model_function(c::kd.AbstractContour, t_f::kd.BranchPoint, h::Vector)
    u_f = kd.get_ref(c, t_f)
    # Eq. (6)
    return u::Vector{Float64} -> begin
        # Transformation u -> v
        v = similar(u)
        v[1] = u_f - u[1]
        v[2:end] = -diff(u)
        # Product of h(v_i)
        reduce(*, [hi(vi) for (hi, vi) in zip(h, v)])
    end
end

"""
    $(TYPEDSIGNATURES)

Make the model function ``p_d(\\mathbf{u}) = \\prod_{i=1}^d h(u_{i-1} - u_i)`` out of an
exponential function ``h(v) = e^{-v/\\tau}``.

``u_0`` is the distance measured along the time contour `c` to the point `t_f`.
"""
function make_exp_model_function(c::kd.AbstractContour,
                                 t_f::kd.BranchPoint,
                                 τ::Real,
                                 d::Int)
    u_f = kd.get_ref(c, t_f)
    # Eq. (6)
    return u::Vector{Float64} -> begin # Transformation u -> v
        # Simplification: \sum_i exp(-v_i / τ) = exp(-(u_f - u[end]) / τ)
        exp(-(u_f - u[end]) / τ)
    end
end

"""
    $(TYPEDSIGNATURES)

Make a transformation from the unit hypercube ``\\mathbf{x}\\in[0, 1]^d`` to a
``d``-dimensional simplex ``u_f > u_1 > u_2 > \\ldots > u_{d-1} > u_d > -\\infty``
induced by the model function
``p_d(\\mathbf{u}) = \\prod_{i=1}^d e^{(u_{i-1} - u_i)/\\tau}``.

The target variables `u_i` are reference values of time points measured as distances along
the contour `c`. ``u_0`` is the distance to the point `t_f`.
"""
function make_exp_trans(c::kd.AbstractContour, t_f::kd.BranchPoint, τ::Real)
    u_f = kd.get_ref(c, t_f)
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
    $(TYPEDSIGNATURES)

Norm of the model function ``p_d(\\mathbf{u}) = \\prod_{i=1}^d e^{(u_{i-1} - u_i)/\\tau}``.
"""
exp_p_norm(τ::Real, d::Int)::Real = τ^d

"""
    $(TYPEDSIGNATURES)

Quasi Monte Carlo integration with warping.

# Parameters
- `f`:      Integrand.
- `init`:   Initial (zero) value of the integral.
- `p`:      Positive model function ``p_d(\\mathbf{u})``.
- `p_norm`: Integral of ``p_d(\\mathbf{u})`` over the ``\\mathbf{u}``-domain.
- `trans`:  Transformation from ``\\mathbf{x}\\in[0, 1]^d`` onto the ``\\mathbf{u}``-domain.
- `seq`:    Quasi-random sequence generator.
- `N`:      The number of points to be taken from the quasi-random sequence.

# Returns
Value of the integral.
"""
function qmc_integral(f, init = zero(typeof(f(trans(0))));
                      p, p_norm, trans, seq, N::Int)
    # Eq. (5)
    res = init
    for i = 1:N
        x = next!(seq)
        u = trans(x)
        f_val = f(u)
        isnothing(f_val) && continue
        res += f_val * (1.0 / p(u))
    end
    (p_norm / N) * res
end

"""
    $(TYPEDSIGNATURES)

Quasi Monte Carlo integration with warping.

This function takes a specified number of valid (non-`nothing`) samples of the integrand.

# Parameters
- `f`:         Integrand.
- `init`:      Initial (zero) value of the integral.
- `p`:         Positive model function ``p_d(\\mathbf{u})``.
- `p_norm`:    Integral of ``p_d(\\mathbf{u})`` over the ``\\mathbf{u}``-domain.
- `trans`:     Transformation from ``\\mathbf{x}\\in[0, 1]^d`` onto the
               ``\\mathbf{u}``-domain.
- `seq`:       Quasi-random sequence generator.
- `N_samples`: The number of samples to be taken.

# Returns
Value of the integral.
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
        isnothing(f_val) && continue
        res += f_val * (1.0 / p(u))
        i += 1
    end
    (p_norm / N) * res
end

"""
Dictionary mapping branches of the Keldysh contour to their unitary direction coefficients
in the complex time plane.
"""
const branch_direction = Dict(
    kd.forward_branch => 1.0,
    kd.backward_branch => -1.0,
    kd.imaginary_branch => -im
)

"""
    $(TYPEDSIGNATURES)

Detect the return type of a function applied to a vector of `Keldysh.BranchPoint`.
"""
function contour_function_return_type(f::Function)
    Base.return_types(f, (Vector{kd.BranchPoint},))[1]
end

"""
    $(TYPEDSIGNATURES)

Evaluate a ``d``-dimensional contour-ordered integral of a function ``f(\\mathbf{t})``,
```math
    \\int_{t_i}^{t_f} dt_1 \\int_{t_i}^{t_1} dt_2 \\ldots \\int_{t_i}^{t_{d-1}} dt_d
    \\ f(t_1, t_2, \\ldots, t_d)
```
using the Sobol sequence for quasi-random sampling and the exponential model function
``p_d(\\mathbf{t}) = \\prod_{i=1}^d e^{(t_{i-1} - t_i)/\\tau}``.

# Parameters
- `f`:    Integrand.
- `d`:    Dimensionality of the integral.
- `c`:    Time contour to integrate over.
- `t_i`:  Starting time point on the contour.
- `t_f`:  Final time point on the contour.
- `init`: Initial (zero) value of the integral.
- `seq`:  Quasi-random sequence generator.
- `τ`:    Decay parameter of the exponential model function.
- `N`:    The number of points to be taken from the quasi-random sequence.

# Returns
Value of the integral.
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
    @boundscheck kd.heaviside(c, t_f, t_i)

    # Model function, its norm and the x -> u transformation
    p_d = make_exp_model_function(c, t_f, τ, d)
    p_d_norm = exp_p_norm(τ, d)
    trans = make_exp_trans(c, t_f, τ)

    u_i = kd.get_ref(c, t_i)

    N_samples::Int = 0

    return (qmc_integral(init,
                         p = p_d,
                         p_norm = p_d_norm,
                         trans = trans,
                         seq = seq,
                         N = N) do refs
        refs[end] < u_i && return nothing # Discard irrelevant reference values
        N_samples += 1
        t_points = [c(r) for r in refs]
        return prod(t -> branch_direction[t.domain], t_points) * f(t_points)
    end, N_samples)
end

"""
    $(TYPEDSIGNATURES)

Evaluate a ``d``-dimensional contour-ordered integral of a function ``f(\\mathbf{t})``,
```math
    \\int_{t_i}^{t_f} dt_1 \\int_{t_i}^{t_1} dt_2 \\ldots \\int_{t_i}^{t_{d-1}} dt_d
    \\ f(t_1, t_2, \\ldots, t_d)
```
using the Sobol sequence for quasi-random sampling and the exponential model function
``p_d(\\mathbf{t}) = \\prod_{i=1}^d e^{(t_{i-1} - t_i)/\\tau}``.

This function evaluates the integrand a specified number of times while discarding any
transformed sequence points that fall outside the integration domain.

# Parameters
- `f`:         Integrand.
- `d`:         Dimensionality of the integral.
- `c`:         Time contour to integrate over.
- `t_i`:       Starting time point on the contour.
- `t_f`:       Final time point on the contour.
- `init`:      Initial (zero) value of the integral.
- `seq`:       Quasi-random sequence generator.
- `τ`:         Decay parameter of the exponential model function.
- `N_samples`: The number of samples to be taken.

# Returns
Value of the integral.
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
    @boundscheck kd.heaviside(c, t_f, t_i)

    # Model function, its norm and the x -> u transformation
    p_d = make_exp_model_function(c, t_f, τ, d)
    p_d_norm = exp_p_norm(τ, d)
    trans = make_exp_trans(c, t_f, τ)

    u_i = kd.get_ref(c, t_i)

    N::Int = 0

    return (qmc_integral_n_samples(init,
                                   p = p_d,
                                   p_norm = p_d_norm,
                                   trans = trans,
                                   seq = seq,
                                   N_samples = N_samples) do refs
        N += 1
        refs[end] < u_i && return nothing # Discard irrelevant reference values
        t_points = [c(r) for r in refs]
        return prod(t -> branch_direction[t.domain], t_points) * f(t_points)
    end, N)
end

"""
    $(TYPEDSIGNATURES)

Evaluate a ``d``-dimensional contour-ordered integral of a function ``f(\\mathbf{t})``,
```math
    \\int_{t_i}^{t_f} dt_1 \\int_{t_i}^{t_1} dt_2 \\ldots \\int_{t_i}^{t_{d-1}} dt_d
    \\ f(t_1, t_2, \\ldots, t_d)
```
using the Sobol sequence for quasi-random sampling and the 'Sort' transform.

# Parameters
- `f`:    Integrand.
- `d`:    Dimensionality of the integral.
- `c`:    Time contour to integrate over.
- `t_i`:  Starting time point on the contour.
- `t_f`:  Final time point on the contour.
- `init`: Initial (zero) value of the integral.
- `seq`:  Quasi-random sequence generator.
- `N`:    The number of points to be taken from the quasi-random sequence.

# Returns
Value of the integral.
"""
function qmc_time_ordered_integral_sort(f,
                                        d::Int,
                                        c::kd.AbstractContour,
                                        t_i::kd.BranchPoint,
                                        t_f::kd.BranchPoint;
                                        init = zero(contour_function_return_type(f)),
                                        seq = SobolSeqWith0(d),
                                        N::Int)
    @boundscheck kd.heaviside(c, t_f, t_i)

    # x -> u transformation
    u_i = kd.get_ref(c, t_i)
    u_f = kd.get_ref(c, t_f)
    ref_diff = u_f - u_i
    trans = x -> u_i .+ sort(x, rev=true) * ref_diff

    return qmc_integral(init,
                        p = u -> 1.0,
                        p_norm = (ref_diff ^ d) / factorial(d),
                        trans = trans,
                        seq = seq,
                        N = N) do refs
        t_points = [c(r) for r in refs]
        return prod(t -> branch_direction[t.domain], t_points) * f(t_points)
    end
end

"""
    $(TYPEDSIGNATURES)

Evaluate a ``d``-dimensional contour-ordered integral of a function ``f(\\mathbf{t})``,
```math
    \\int_{t_i}^{t_f} dt_1 \\int_{t_i}^{t_1} dt_2 \\ldots \\int_{t_i}^{t_{d-1}} dt_d
    \\ f(t_1, t_2, \\ldots, t_d)
```
using the Sobol sequence for quasi-random sampling and the 'Root' transform.

# Parameters
- `f`:    Integrand.
- `d`:    Dimensionality of the integral.
- `c`:    Time contour to integrate over.
- `t_i`:  Starting time point on the contour.
- `t_f`:  Final time point on the contour.
- `init`: Initial (zero) value of the integral.
- `seq`:  Quasi-random sequence generator.
- `N`:    The number of points to be taken from the quasi-random sequence.

# Returns
Value of the integral.
"""
function qmc_time_ordered_integral_root(f,
                                        d::Int,
                                        c::kd.AbstractContour,
                                        t_i::kd.BranchPoint,
                                        t_f::kd.BranchPoint;
                                        init = zero(contour_function_return_type(f)),
                                        seq = SobolSeqWith0(d),
                                        N::Int)
    @boundscheck kd.heaviside(c, t_f, t_i)

    # x -> u transformation
    u_i = kd.get_ref(c, t_i)
    u_f = kd.get_ref(c, t_f)
    ref_diff = u_f - u_i

    trans = x -> begin
        u = Vector{Float64}(undef, d)
        u[1] = x[1] ^ (1.0 / d)
        for s = 2:d
            u[s] = u[s - 1] * (x[s] ^ (1.0 / (d - s + 1)))
        end
        return u_i .+ u * ref_diff
    end

    return qmc_integral(init,
                        p = u -> 1.0,
                        p_norm = (ref_diff ^ d) / factorial(d),
                        trans = trans,
                        seq = seq,
                        N = N) do refs
        t_points = [c(r) for r in refs]
        return prod(t -> branch_direction[t.domain], t_points) * f(t_points)
    end
end

"""
    $(TYPEDSIGNATURES)

Evaluate an inchworm-type contour-ordered integral of a function ``f(\\mathbf{t})``,
```math
\\int_{t_w}^{t_f} dt_1
\\int_{t_w}^{t_1} dt_2 \\ldots
\\int_{t_w}^{t_{d_{after}-1}} dt_{d_{after}}
\\int_{t_i}^{t_w} dt_{d_{after}+1} \\ldots
\\int_{t_i}^{t_{d-2}} dt_{d-1}
\\int_{t_i}^{t_{d-1}} dt_d\\
    f(t_1, t_2, \\ldots, t_d)
```

using the Sobol sequence for quasi-random sampling and a two-piece 'Root' transform.
The total dimension of the domain `d` is the sum of the amounts of integration
variables in the 'before' (``d_\\text{before}``) and the after (``d_\\text{after}``)
components.

# Parameters
- `f`:        Integrand.
- `d_before`: Dimensionality of the before-``t_w`` component of the integration domain.
- `d_after`:  Dimensionality of the after-``t_w`` component of the integration domain.
- `c`:        Time contour to integrate over.
- `t_i`:      Starting time point on the contour.
- `t_w`:      'Worm' time point on the contour separating the 'before' and 'after' parts.
- `t_f`:      Final time point on the contour.
- `init`:     Initial (zero) value of the integral.
- `seq`:      Quasi-random sequence generator.
- `N`:        The number of samples to be taken.

# Returns
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
    @boundscheck kd.heaviside(c, t_w, t_i)
    @boundscheck kd.heaviside(c, t_f, t_w)
    @boundscheck d_before >= 0
    @boundscheck d_after >= 1

    u_i = kd.get_ref(c, t_i)
    u_w = kd.get_ref(c, t_w)
    u_f = kd.get_ref(c, t_f)

    ref_diff_wi = u_w - u_i
    ref_diff_fw = u_f - u_w

    d = d_before + d_after

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

    return qmc_integral(init,
                        p = u -> 1.0,
                        p_norm = p_norm,
                        trans = trans,
                        seq = seq,
                        N = N) do refs
        t_points = [c(r) for r in refs]
        return prod(t -> branch_direction[t.domain], t_points) * f(t_points)
    end
end

end # module qmc_integrate
