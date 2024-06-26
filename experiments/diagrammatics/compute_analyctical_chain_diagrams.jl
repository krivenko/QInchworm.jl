using InverseLaplace: Talbot
using Printf

using HDF5; h5 = HDF5

"""
Make the Laplace image of d_n(α, x) w.r.t. x
"""
function make_Ldn(n, α)
    return s -> factorial(big(2n)) / (s^2 * (s + α)^n * (s + 2α)^(n-1))
end

# Multi-precision Laplace transform inversion
# Abate, J. and Valkó, P.P.
# International Journal for Numerical Methods in Engineering, Vol. 60 (Iss. 5-7)  2004  pp 979–993
# https://doi.org/10.1002/nme.995

#setprecision(BigFloat, 4000)
setprecision(BigFloat, 1000)

V = 1
ϵ = 1
β = 0.0

α = β*ϵ

label = "log"
#nmax = 13
nmax = 13
ns = Array{Int64}(2 .^ (0:nmax))

#label = "linear"
#nmax = 128
#ns = Array{Int64}(1:nmax)

@show ns

Ds = Array{Float64}(undef, 0)

for n in ns
    d_n = Talbot(make_Ldn(n, α), max(4n, 100))(big(1.0))
    #d_n = Talbot(make_Ldn(n, ϵ), max(4n, 100))(big(β)) / big(β)^(2n)
    @show n, nmax, Float64(d_n)
    push!(Ds, d_n)
end

#Ds = Array{Float64}([ Talbot(make_Ldn(n, α), max(2n, 100))(big(1.0)) for n in ns ])

@show Ds
#exit()

filename = "data_analytic_chain_integral_beta_$(β)_V_$(V)_n_max_$(nmax)_invlap_$(label).h5"

@show filename
h5.h5open(filename, "w") do fid
    g = h5.create_group(fid, "data")
    
    h5.attributes(g)["epsilon"] = ϵ
    h5.attributes(g)["V"] = V
    h5.attributes(g)["beta"] = β
    
    g["ns"] = ns
    g["Ds"] = Ds
end
