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

[^1]: Integration domain transformations based on product model functions are described in
      "Quantum Quasi-Monte Carlo Technique for Many-Body Perturbative Expansions",
      M. Maček, P. T. Dumitrescu, C. Bertrand, B.Triggs, O. Parcollet, and X. Waintal,
      Phys. Rev. Lett. 125, 047702 (2020).

[^2]: Integration domain transformations `Sort` and `Root` are defined in
      "Transforming low-discrepancy sequences from a cube to a simplex",
      T. Pillards and R. Cools, J. Comput. Appl. Math. 174, 29 (2005).
"""
module qmc_integrate

using DocStringExtensions

using MPI: MPI

using Keldysh; kd = Keldysh

using QInchworm.scrambled_sobol: ScrambledSobolSeq, next!

#
# AbstractDomainTransform
#

"""
    $(TYPEDEF)

Abstract domain transformation ``[0, 1]^d \\mapsto \\mathscr{D}``.
"""
abstract type AbstractDomainTransform{D} end

"""
    $(TYPEDSIGNATURES)

Return the number of dimensions ``d`` of a domain transformation
``[0, 1]^d \\mapsto \\mathscr{D}``.
"""
Base.ndims(::AbstractDomainTransform{D}) where D = D

#
# ExpModelFunctionTransform
#

"""
    $(TYPEDEF)

Domain transformation
```math
\\mathbf{x}\\in[0,1]^d \\mapsto
    \\{\\mathbf{u}: u_f \\geq u_1 \\geq u_2 \\geq \\ldots \\geq u_d > -\\infty\\}
```
induced by the implicit variable change
```math
x_n(v_n) = \\frac{\\int_0^{v_n} d\\bar v_n h(\\bar v_n)}
                 {\\int_0^\\infty d\\bar v_n h(\\bar v_n)},
\\quad
v_n = \\left\\{
\\begin{array}{ll}
u_f - u_1, &n=1,\\\\
u_{n-1} - u_n, &n>1,
\\end{array}
\\right.
```
where ``h(v)`` is an exponential model function parametrized by decay rate ``\\tau``,
``h(v) = e^{-v/\\tau}``.

The corresponding Jacobian is ``J(\\mathbf{u}) = \\tau^d / e^{-(u_f - u_d) / \\tau}``.

# Fields
$(TYPEDFIELDS)
"""
struct ExpModelFunctionTransform{D} <: AbstractDomainTransform{D}
    "Upper bound of the transformed domain ``u_f``"
    u_f::Float64
    "Decay rate parameter of the exponential model function"
    τ::Float64
end

"""
$(TYPEDSIGNATURES)

Make an [`ExpModelFunctionTransform`](@ref) object suitable for time contour integration over
the domain
```math
\\{(t_1, \\ldots, t_d) \\in \\mathcal{C}^d:
    t_f \\succeq t_1 \\succeq t_2 \\succeq \\ldots \\succeq t_d \\succeq
    \\text{starting point of }\\mathcal{C}\\}
```
**N.B.** [`ExpModelFunctionTransform`](@ref) describes an infinite domain where integration
variables ``u_n`` can approach ``-\\infty``. Negative values of ``u_n`` cannot be mapped
onto time points on ``\\mathcal{C}`` and will be discarded by [`contour_integral()`](@ref).

# Parameters
- `d`:   Number of dimensions ``d``.
- `c`:   Time contour ``\\mathcal{C}``.
- `t_f`: Upper bound ``t_f``.
- `τ`:   Decay rate parameter ``\\tau``.
"""
function ExpModelFunctionTransform(d::Integer,
                                   c::kd.AbstractContour,
                                   t_f::kd.BranchPoint,
                                   τ::Real)
    return ExpModelFunctionTransform{d}(kd.get_ref(c, t_f), τ)
end

"""
    $(TYPEDSIGNATURES)

Return the function ``\\mathbf{u}(\\mathbf{x})`` corresponding to a given
[`ExpModelFunctionTransform`](@ref) object `t`.
"""
function make_trans_f(t::ExpModelFunctionTransform)::Function
    d = ndims(t)
    u = Vector{Float64}(undef, d)
    return x -> begin
        u[1] = t.u_f + t.τ*log(1 - x[1])
        for i = 2:length(x)
            u[i] = u[i-1] + t.τ*log(1 - x[i])
        end
        return u
    end
end

"""
    $(TYPEDSIGNATURES)

Return the Jacobian ``J(\\mathbf{u}) =
\\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|`` corresponding to a given
[`ExpModelFunctionTransform`](@ref) object `t`.
"""
function make_jacobian_f(t::ExpModelFunctionTransform)::Function
    d = ndims(t)
    norm = t.τ^d
    return u -> begin
        # Simplification: \sum_i exp(-v_i / τ) = exp(-(u_f - u[end]) / τ)
        return norm / exp(-(t.u_f - u[end]) / t.τ)
    end
end

#
# RootTransform
#

"""
    $(TYPEDEF)

Hypercube-to-simplex domain transformation
```math
\\mathbf{x}\\in[0,1]^d \\mapsto
    \\{\\mathbf{u}: u_f \\geq u_1 \\geq u_2 \\geq \\ldots \\geq u_d \\geq u_i\\}
```
induced by the `Root` mapping[^2]
```math
\\left\\{
\\begin{array}{ll}
\\tilde u_1 &= x_1^{1/d},\\\\
\\tilde u_2 &= \\tilde u_1 x_2^{1/(d-1)},\\\\
&\\ldots\\\\
\\tilde u_d &= \\tilde u_{d-1} x_d,
\\end{array}\\right.
```
```math
\\mathbf{u} = u_i + (u_f - u_i) \\tilde{\\mathbf{u}}.
```

The corresponding Jacobian is ``J(\\mathbf{u}) = (u_f - u_i)^d / d!``.

# Fields
$(TYPEDFIELDS)
"""
struct RootTransform{D} <: AbstractDomainTransform{D}
    "Lower bound of the transformed domain ``u_i``"
    u_i::Float64
    "Difference ``u_f - u_i``"
    u_diff::Float64
end

"""
    $(TYPEDSIGNATURES)

Make a [`RootTransform`](@ref) object suitable for time contour integration over the domain
```math
\\{(t_1, \\ldots, t_d) \\in \\mathcal{C}^d:
    t_f \\succeq t_1 \\succeq t_2 \\succeq \\ldots \\succeq t_d \\succeq t_i\\}
```

# Parameters
- `d`:   Number of dimensions ``d``.
- `c`:   Time contour ``\\mathcal{C}``.
- `t_i`: Lower bound ``t_i``.
- `t_f`: Upper bound ``t_f``.
"""
function RootTransform(d::Integer,
                       c::kd.AbstractContour,
                       t_i::kd.BranchPoint,
                       t_f::kd.BranchPoint)
    @boundscheck kd.heaviside(c, t_f, t_i)
    u_i = kd.get_ref(c, t_i)
    return RootTransform{d}(u_i, kd.get_ref(c, t_f) - u_i)
end

"""
    $(TYPEDSIGNATURES)

Return the function ``\\mathbf{u}(\\mathbf{x})`` corresponding to a given
[`RootTransform`](@ref) object `t`.
"""
function make_trans_f(t::RootTransform)::Function
    d = ndims(t)
    u = Vector{Float64}(undef, d)
    return x -> begin
        u[1] = x[1] ^ (1.0 / d)
        for i = 2:d
            u[i] = u[i - 1] * (x[i] ^ (1.0 / (d - i + 1)))
        end
        return t.u_i .+ u * t.u_diff
    end
end

"""
    $(TYPEDSIGNATURES)

Return the Jacobian ``J(\\mathbf{u}) =
\\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|`` corresponding to a given
[`RootTransform`](@ref) object `t`.
"""
function make_jacobian_f(t::RootTransform)::Function
    d = ndims(t)
    simplex_volume = (t.u_diff ^ d) / factorial(big(d))
    return u -> simplex_volume
end

#
# SortTransform
#

"""
    $(TYPEDEF)

Hypercube-to-simplex domain transformation
```math
\\mathbf{x}\\in[0,1]^d \\mapsto
    \\{\\mathbf{u}: u_f \\geq u_1 \\geq u_2 \\geq \\ldots \\geq u_d \\geq u_i\\}
```
induced by the `Sort` mapping[^2]
```math
\\mathbf{u} = u_i + (u_f - u_i) \\mathrm{sort}(x_1, \\ldots, x_d).
```

The corresponding Jacobian is ``J(\\mathbf{u}) = (u_f - u_i)^d / d!``.

# Fields
$(TYPEDFIELDS)
"""
struct SortTransform{D} <: AbstractDomainTransform{D}
    "Lower bound of the transformed domain ``u_i``"
    u_i::Float64
    "Difference ``u_f - u_i``"
    u_diff::Float64
end

"""
    $(TYPEDSIGNATURES)

Make a [`SortTransform`](@ref) object suitable for time contour integration over the domain
```math
\\{(t_1, \\ldots, t_d) \\in \\mathcal{C}^d:
    t_f \\succeq t_1 \\succeq t_2 \\succeq \\ldots \\succeq t_d \\succeq t_i\\}
```

# Parameters
- `d`:   Number of dimensions ``d``.
- `c`:   Time contour ``\\mathcal{C}``.
- `t_i`: Lower bound ``t_i``.
- `t_f`: Upper bound ``t_f``.
"""
function SortTransform(d::Integer,
                       c::kd.AbstractContour,
                       t_i::kd.BranchPoint,
                       t_f::kd.BranchPoint)
    @boundscheck kd.heaviside(c, t_f, t_i)
    u_i = kd.get_ref(c, t_i)
    return SortTransform{d}(u_i, kd.get_ref(c, t_f) - u_i)
end

"""
    $(TYPEDSIGNATURES)

Return the function ``\\mathbf{u}(\\mathbf{x})`` corresponding to a given
[`SortTransform`](@ref) object `t`.
"""
function make_trans_f(t::SortTransform)::Function
    return x -> t.u_i .+ sort(x, rev=true) * t.u_diff
end

"""
    $(TYPEDSIGNATURES)

Return the Jacobian ``J(\\mathbf{u}) =
\\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|`` corresponding to a given
[`SortTransform`](@ref) object `t`.
"""
function make_jacobian_f(t::SortTransform)::Function
    d = ndims(t)
    simplex_volume = (t.u_diff ^ d) / factorial(d)
    return u -> simplex_volume
end

#
# DoubleSimplexRootTransform
#

"""
    $(TYPEDEF)
Hypercube-to-double-simplex domain transformation

```math
\\mathbf{x}\\in[0,1]^d \\mapsto
    \\{\\mathbf{u}: u_f \\geq u_1 \\geq u_2 \\geq \\ldots \\geq u_{d^>} \\geq u_w \\geq
                    u_{d^>+1} \\geq \\ldots \\geq u_d \\geq u_i\\}
```
induced by the `Root` mapping[^2] applied independently to two sets of variables
``\\{x_1, \\ldots, x_{d^>} \\}`` and ``\\{x_{d^>+1}, \\ldots, x_d \\}``
(cf. [`RootTransform`](@ref)).

The corresponding Jacobian is
``J(\\mathbf{u}) = \\frac{(u_w - u_i)^{d^<}}{d^<!} \\frac{(u_f - u_w)^{d^>}}{d^>!}``.

Here, ``d^<`` and ``d^>`` denote numbers of variables in the 'lesser' and 'greater' simplex
respectively, and ``d = d^< + d^>``. The two simplices are separated by a fixed boundary
located at ``u_w``.

# Fields
$(TYPEDFIELDS)
"""
struct DoubleSimplexRootTransform{D} <: AbstractDomainTransform{D}
    "Number of variables in the 'lesser' simplex, ``d^<``"
    d_lesser::Int
    "Number of variables in the 'greater' simplex, ``d^>``"
    d_greater::Int
    "Lower bound of the transformed domain ``u_i``"
    u_i::Float64
    "Boundary ``u_w`` separating the 'lesser' and the 'greater' simplices"
    u_w::Float64
    "Difference ``u_w - u_i``"
    u_diff_wi::Float64
    "Difference ``u_f - u_w``"
    u_diff_fw::Float64
end

"""
    $(TYPEDSIGNATURES)

Make a [`DoubleSimplexRootTransform`](@ref) object suitable for time contour integration
over the domain
```math
\\{(t_1, \\ldots, t_d) \\in \\mathcal{C}^d:
    t_f \\succeq t_1 \\succeq t_2 \\succeq \\ldots \\succeq t_{d_\\text{after}} \\succeq t_w
    \\succeq t_{d_\\text{after}+1} \\succeq \\ldots \\succeq t_d \\succeq t_i\\},
```
where ``d = d_\\text{before} + d_\\text{after}``.

# Parameters
- `d_before`: Number of time variables in the 'before' region, ``d_\\text{before}``.
- `d_after`:  Number of time variables in the 'after' region, ``d_\\text{after}``.
- `c`:        Time contour ``\\mathcal{C}``.
- `t_i`:      Lower bound ``t_i``.
- `t_w`:      Boundary point ``t_w``.
- `t_f`:      Upper bound ``t_f``.
"""
function DoubleSimplexRootTransform(d_before::Int,
                                    d_after::Int,
                                    c::kd.AbstractContour,
                                    t_i::kd.BranchPoint,
                                    t_w::kd.BranchPoint,
                                    t_f::kd.BranchPoint)
    @boundscheck kd.heaviside(c, t_w, t_i)
    @boundscheck kd.heaviside(c, t_f, t_w)
    @boundscheck d_before >= 0
    @boundscheck d_after >= 1

    d = d_before + d_after

    u_i = kd.get_ref(c, t_i)
    u_w = kd.get_ref(c, t_w)
    u_diff_wi = u_w - u_i
    u_diff_fw = kd.get_ref(c, t_f) - u_w

    return DoubleSimplexRootTransform{d}(d_before, d_after, u_i, u_w, u_diff_wi, u_diff_fw)
end

"""
    $(TYPEDSIGNATURES)

Return the function ``\\mathbf{u}(\\mathbf{x})`` corresponding to a given
[`DoubleSimplexRootTransform`](@ref) object `t`.
"""
function make_trans_f(t::DoubleSimplexRootTransform)::Function
    d = ndims(t)
    u = Vector{Float64}(undef, d)
    return x -> begin
        # Points in the 'greater' region
        u[1] = x[1] ^ (1.0 / t.d_greater)
        for i = 2:t.d_greater
            u[i] = u[i - 1] * (x[i] ^ (1.0 / (t.d_greater - i + 1)))
        end
        u[1:t.d_greater] *= t.u_diff_fw
        u[1:t.d_greater] .+= t.u_w

        t.d_lesser == 0 && return u

        # Points in the 'lesser' region
        u[t.d_greater + 1] = x[t.d_greater + 1] ^ (1.0 / t.d_lesser)
        for i = (t.d_greater + 2):d
            u[i] = u[i - 1] * (x[i] ^ (1.0 / (d - i + 1)))
        end
        u[t.d_greater+1:d] *= t.u_diff_wi
        u[t.d_greater+1:d] .+= t.u_i

        return u
    end
end

"""
    $(TYPEDSIGNATURES)

Return the Jacobian ``J(\\mathbf{u}) =
\\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|`` corresponding to a given
[`DoubleSimplexRootTransform`](@ref) object `t`.
"""
function make_jacobian_f(t::DoubleSimplexRootTransform)::Function
    before_simplex_volume = (t.u_diff_wi ^ t.d_lesser) / factorial(t.d_lesser)
    after_simplex_volume = (t.u_diff_fw ^ t.d_greater) / factorial(t.d_greater)
    volume = before_simplex_volume * after_simplex_volume
    return u -> volume
end

#
# Integration
#

"""
    $(TYPEDSIGNATURES)

Compute a quasi Monte Carlo estimate of a ``d``-dimensional integral
```math
F = \\int_\\mathscr{D} d^d \\mathbf{u}\\ f(\\mathbf{u}).
```
The domain ``\\mathscr{D}`` is defined by a variable change ``\\mathbf{u} =
\\mathbf{u}(\\mathbf{x}): [0,1]^d \\mapsto \\mathscr{D}``,
```math
F = \\int_{[0,1]^d} d^d\\mathbf{x}\\ f(\\mathbf{u}(\\mathbf{x}))
    \\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|
\\approx \\frac{1}{N} \\sum_{i=1}^N
    f(\\mathbf{u}(\\mathbf{x}_i)) J(\\mathbf{u}(\\mathbf{x}_i)).
```

# Parameters
- `f`:          Integrand ``f(\\mathbf{u})``.
- `init`:       Initial (zero) value used in qMC summation.
- `trans_f`:    Domain transformation function ``\\mathbf{u}(\\mathbf{x})``.
- `jacobian_f`: Jacobian ``J(\\mathbf{u}) =
                \\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|``.
- `seq`:        Quasi-random sequence generator.
- `N`:          Number of points to be taken from the quasi-random sequence.

# Returns
Estimated value of the integral.
"""
function qmc_integral(f, init = zero(f(trans_f(.0))); trans_f, jacobian_f, seq, N::Int)
    res = init
    for i = 1:N
        x = next!(seq)
        u = trans_f(x)
        f_val = f(u)
        isnothing(f_val) && continue
        res += f_val * jacobian_f(u)
    end
    return (1 / N) * res
end

"""
    $(TYPEDSIGNATURES)

Compute a quasi Monte Carlo estimate of a ``d``-dimensional integral
```math
F = \\int_\\mathscr{D} d^d \\mathbf{u}\\ f(\\mathbf{u}).
```
The domain ``\\mathscr{D}`` is defined by a variable change ``\\mathbf{u} =
\\mathbf{u}(\\mathbf{x}): [0,1]^d \\mapsto \\mathscr{D}``,
```math
F = \\int_{[0,1]^d} d^d\\mathbf{x}\\ f(\\mathbf{u}(\\mathbf{x}))
    \\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|
\\approx \\frac{1}{N} \\sum_{i=1}^N
    f(\\mathbf{u}(\\mathbf{x}_i)) J(\\mathbf{u}(\\mathbf{x}_i)).
```

Unlike [`qmc_integral()`](@ref), this function performs qMC summation until a given number
of valid (non-`nothing`) samples of ``f(\\mathbf{u})`` are taken.

# Parameters
- `f`:          Integrand ``f(\\mathbf{u})``.
- `init`:       Initial (zero) value used in qMC summation.
- `trans_f`:    Domain transformation function ``\\mathbf{u}(\\mathbf{x})``.
- `jacobian_f`: Jacobian ``J(\\mathbf{u}) =
                \\left|\\frac{\\partial\\mathbf{u}}{\\partial\\mathbf{x}}\\right|``.
- `seq`:        Quasi-random sequence generator.
- `N_samples`:  Number of valid samples of ``f(\\mathbf{u})`` to be taken.

# Returns
Estimated value of the integral.
"""
function qmc_integral_n_samples(f,
                                init = zero(f(trans_f(.0)));
                                trans_f,
                                jacobian_f,
                                seq,
                                N_samples::Int)
    res = init
    i = 1
    N = 0
    while i <= N_samples
        N += 1
        x = next!(seq)
        u = trans_f(x)
        f_val = f(u)
        isnothing(f_val) && continue
        res += f_val * jacobian_f(u)
        i += 1
    end
    return (1 / N) * res
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

Compute a quasi Monte Carlo estimate of a contour integral over
a ``d``-dimensional domain ``\\mathscr{D}``.

# Parameters
- `f`:    Integrand.
- `c`:    Time contour to integrate over.
- `dt`:   Domain transformation ``[0, 1]^d \\mapsto \\mathscr{D}``.
- `init`: Initial (zero) value of the integral.
- `seq`:  Quasi-random sequence generator.
- `N`:    The number of points to be taken from the quasi-random sequence.

# Returns
Estimated value of the contour integral.
"""
function contour_integral(f,
                          c::kd.AbstractContour,
                          dt::AbstractDomainTransform;
                          init = zero(contour_function_return_type(f)),
                          seq = ScrambledSobolSeq(ndims(dt)),
                          N::Int)
    return qmc_integral(init,
                        trans_f=make_trans_f(dt),
                        jacobian_f=make_jacobian_f(dt),
                        seq=seq,
                        N=N) do refs
        all(refs .>= 0.0) || return nothing # Discard irrelevant reference values
        t_points = c.(refs)
        return prod(t -> branch_direction[t.domain], t_points) * f(t_points)
    end
end

"""
    $(TYPEDSIGNATURES)

Compute a quasi Monte Carlo estimate of a contour integral over
a ``d``-dimensional domain ``\\mathscr{D}``.

Unlike [`contour_integral()`](@ref), this function performs qMC summation until a given
number of valid (non-`nothing`) samples of the integrand are taken.

# Parameters
- `f`:         Integrand.
- `c`:         Time contour to integrate over.
- `dt`:        Domain transformation ``[0, 1]^d \\mapsto \\mathscr{D}``.
- `init`:      Initial (zero) value of the integral.
- `seq`:       Quasi-random sequence generator.
- `N_samplex`: Number of valid samples of the integrand to be taken.

# Returns
- Estimated value of the contour integral.
- The total number of points taken from the quasi-random sequence.
"""
function contour_integral_n_samples(f,
                                    c::kd.AbstractContour,
                                    dt::AbstractDomainTransform;
                                    init = zero(contour_function_return_type(f)),
                                    seq = ScrambledSobolSeq(ndims(dt)),
                                    N_samples::Int)
    N::Int = 1
    return (qmc_integral_n_samples(init,
                                   trans_f=make_trans_f(dt),
                                   jacobian_f=make_jacobian_f(dt),
                                   seq=seq,
                                   N_samples=N_samples) do refs
        N += 1
        all(refs .>= 0.0) || return nothing # Discard irrelevant reference values
        t_points = c.(refs)
        return prod(t -> branch_direction[t.domain], t_points) * f(t_points)
    end, N)
end

end # module qmc_integrate
