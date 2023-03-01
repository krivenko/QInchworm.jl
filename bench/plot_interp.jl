using PyPlot; plt = PyPlot

using Interpolations: BSplineInterpolation,
                      interpolate,
                      scale,
                      BSpline,
                      Cubic,
                      Quadratic,
                      Linear,
                      Line,
                      Natural,
                      Free,
                      Flat,
                      FlatTest,
                      OnGrid

β = 1.0
ϵ = 2.0

#g = τ -> τ
g = τ -> exp.(-ϵ*τ) ./ ( 1 + exp(-β*ϵ) )
knots = LinRange(0, β, 11)

get_linear = knots -> scale(interpolate(g(knots), BSpline(Linear())), knots)
#get_quadr = knots -> scale(interpolate(g(knots), BSpline(Quadratic(Line(OnGrid())))), knots)
#get_cubic = knots -> scale(interpolate(g(knots), BSpline(Cubic(Line(OnGrid())))), knots)
get_quadr = knots -> scale(interpolate(g(knots), BSpline(Quadratic(FlatTest(OnGrid())))), knots)
get_cubic = knots -> scale(interpolate(g(knots), BSpline(Cubic(FlatTest(OnGrid())))), knots)

linear = get_linear(knots)
quadr = get_quadr(knots)
cubic = get_cubic(knots)

τ = 0.0:β/100000:β

@show knots

plt.figure(figsize=(8, 8))
subp = [2, 2, 1]

plt.subplot(subp...); subp[end] += 1
plt.title("ϵ = $ϵ, β = $β")
plt.plot(knots, g(knots), ".")
plt.plot(τ, g(τ), "-", label=raw"$g(τ) = e^{-ϵτ}/(1+e^{-βϵ})$")
plt.plot(τ, linear(τ), "-", label="linear")
plt.plot(τ, quadr(τ), "-", label="quadr")
plt.plot(τ, cubic(τ), "-", label="cubic")
plt.legend(loc="best")
plt.xlabel(raw"$τ$")
#plt.xlim([0., 0.06])
#plt.ylim([0., 0.06])

plt.subplot(subp...); subp[end] += 1
plt.plot(knots, g(knots), ".")
plt.plot(τ, g(τ), "-", label=raw"$g(τ) = e^{-ϵτ}/(1+e^{-βϵ})$")
plt.plot(τ, linear(τ), "-", label="linear")
plt.plot(τ, quadr(τ), "-", label="quadr")
plt.plot(τ, cubic(τ), "-", label="cubic")
plt.legend(loc="best")
plt.xlabel(raw"$τ$")
plt.semilogy([], [])

plt.subplot(subp...); subp[end] += 1
linear_err = []
quadr_err = []
cubic_err = []
n_vec = []
for e in 3:12
    n = 2^e
    @show n
    knots = LinRange(0, β, n)

    linear = get_linear(knots)
    err = maximum(abs.(g(τ) - linear(τ)))
    push!(linear_err, err)

    quard = get_quadr(knots)
    err = maximum(abs.(g(τ) - quard(τ)))
    push!(quadr_err, err)

    cubic = get_cubic(knots)
    err = maximum(abs.(g(τ) - cubic(τ)))
    push!(cubic_err, err)


    push!(n_vec, n)

end
@show n_vec
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
#plt.xlim([1e0, 1e6])

plt.subplot(subp...); subp[end] += 1
c1 = plt.plot([], [], label="linear error")[1].get_color()
c2 = plt.plot([], [], label="quadr error")[1].get_color()
c3 = plt.plot([], [], label="cubic error")[1].get_color()

#for n in n_vec[1:3:end]
for n in [16, 32]
    knots = LinRange(0, β, n)
    linear = get_linear(knots)
    quadr = get_quadr(knots)
    cubic = get_cubic(knots)

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
