# QInchworm.jl
#
# Copyright (C) 2021-2026 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Author: Hugo U. R. Strand, Igor Krivenko

using PyPlot; plt = PyPlot

using Interpolations: interpolate,
                      scale,
                      BSpline,
                      Linear,
                      Quadratic,
                      Cubic,
                      Flat,
                      OnGrid

β = 1.0
ϵ = 2.0

g = τ -> exp.(-ϵ*τ) ./ (1 + exp(-β*ϵ))
knots = LinRange(0, β, 11)

get_linear = knots -> scale(interpolate(g(knots), BSpline(Linear())), knots)
get_quadr = knots -> scale(interpolate(g(knots), BSpline(Quadratic(Flat(OnGrid())))), knots)
get_cubic = knots -> scale(interpolate(g(knots), BSpline(Cubic(Flat(OnGrid())))), knots)

linear = get_linear(knots)
quadr = get_quadr(knots)
cubic = get_cubic(knots)

τ = 0.0:β/100000:β

plt.figure(figsize=(8, 8))
subp = [2, 2, 1]

plt.subplot(subp...); subp[end] += 1
plt.title("ϵ = $ϵ, β = $β")
plt.plot(knots, g(knots), ".")
plt.plot(τ, g(τ), "-", label=raw"$g(τ) = e^{-\epsilon\tau}/(1+e^{-\beta\epsilon})$")
plt.plot(τ, linear(τ), "-", label="linear")
plt.plot(τ, quadr(τ), "-", label="quadr")
plt.plot(τ, cubic(τ), "-", label="cubic")
plt.legend(loc="best")
plt.xlabel(raw"$τ$")

plt.subplot(subp...); subp[end] += 1
plt.plot(knots, g(knots), ".")
plt.plot(τ, g(τ), "-", label=raw"$g(\tau) = e^{-\epsilon\tau}/(1+e^{-\beta\epsilon})$")
plt.plot(τ, linear(τ), "-", label="linear")
plt.plot(τ, quadr(τ), "-", label="quadr")
plt.plot(τ, cubic(τ), "-", label="cubic")
plt.legend(loc="best")
plt.xlabel(raw"$\tau$")
plt.semilogy([], [])

plt.subplot(subp...); subp[end] += 1
linear_err = []
quadr_err = []
cubic_err = []
n_vec = []
for e in 3:12
    n = 2^e
    local knots = LinRange(0, β, n)

    local linear = get_linear(knots)
    err = maximum(abs.(g(τ) - linear(τ)))
    push!(linear_err, err)

    local quadr = get_quadr(knots)
    err = maximum(abs.(g(τ) - quadr(τ)))
    push!(quadr_err, err)

    local cubic = get_cubic(knots)
    err = maximum(abs.(g(τ) - cubic(τ)))
    push!(cubic_err, err)

    push!(n_vec, n)
end
@show n_vec
@show quadr_err
@show linear_err
@show cubic_err

plt.plot(n_vec, linear_err, "-+", label="linear max error")
plt.plot(n_vec, quadr_err, "-+", label="quadr max error")
plt.plot(n_vec, cubic_err, "-x", label="cubic max error")
plt.legend(loc="best")
plt.xlabel("N interpolation points")
plt.loglog([], [])
plt.axis("image")
plt.grid(true)

plt.subplot(subp...); subp[end] += 1
c1 = plt.plot([], [], label="linear error")[1].get_color()
c2 = plt.plot([], [], label="quadr error")[1].get_color()
c3 = plt.plot([], [], label="cubic error")[1].get_color()

for n in [16, 32]
    local knots = LinRange(0, β, n)
    local linear = get_linear(knots)
    local quadr = get_quadr(knots)
    local cubic = get_cubic(knots)

    plt.plot(τ, abs.(g(τ)-linear(τ)), "-", color=c1, alpha=0.75, zorder=10)
    plt.plot(τ, abs.(g(τ)-quadr(τ)), "-", color=c2, alpha=0.75, zorder=20)
    plt.plot(τ, abs.(g(τ)-cubic(τ)), "-", color=c3, alpha=0.75, zorder=20)
end

plt.legend(loc="best")
plt.xlabel(raw"$τ$")
plt.semilogy([], [])
plt.ylim(bottom=1e-18, top=1)
plt.grid(true)

plt.tight_layout()
plt.savefig("figure_interpolation_scaling.pdf")

plt.show()
