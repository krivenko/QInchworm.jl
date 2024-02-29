# QInchworm.jl
#
# Copyright (C) 2021-2024 I. Krivenko, H. U. R. Strand and J. Kleinhenz
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
# Authors: Igor Krivenko

# Functions defined in this file return `Keldysh.SingularDOS` objects corresponding to
# components of the Nambu Green's function
#
#   G(z) = \int_{-\infty}^{+\infty} \frac{1}{z - \hat H(ϵ)} ρ(ϵ) dϵ,
#
# where \hat H(ϵ) = [ϵ Δ; Δ -ϵ] and ρ(ϵ) is a flat density of states,
# ρ(ϵ) = \frac{1}{2D} θ(D - |ϵ|).

using Keldysh; kd = Keldysh

raw"""
Return a `Keldysh.SingularDOS` object corresponding to the normal component of G(z), i.e.

    -\frac{1}{\pi} \Im[G_{11}(ϵ)].

# Parameters
- `D`: Half-bandwidth of the flat normal state DOS ρ(ϵ).
- `Δ`: Superconducting gap.
"""
function normal_dos(; D=5.0, Δ=1.0)
    @assert Δ > 0
    @assert D > Δ
    ωmax = sqrt(D^2 + Δ^2)
    xmax = ωmax / Δ
    sing_integral = (3 + xmax) / (4 * sqrt(2 * (1 + xmax)))

    kd.SingularDOS(-ωmax, ωmax,
        ω -> begin
            if abs(ω) <= Δ # ω in the superconducting gap
                return .0
            else
                x = ω / Δ
                r = abs(x) / sqrt(x^2 - 1)

                if x > 0
                    rp = sqrt(2 * (x - 1))
                    # The second term regularizes the derivative
                    r -= 1 / rp + (3 / 8) * rp
                else
                    rm = sqrt(-2 * (x + 1))
                    # The second term regularizes the derivative
                    r -= 1 / rm + (3 / 8) * rm
                end

                return r / (2D)
            end
        end,
        [
        kd.DOSSingularity(-Δ,
                          ω -> begin
                              ω >= -Δ && return 0
                              x = ω / Δ
                              rm = sqrt(-2 * (x + 1))
                              # The second term regularizes the derivative
                              return (1.0 / rm + (3 / 8) * rm) / (2D)
                          end,
                          sing_integral
                         ),
        kd.DOSSingularity(Δ,
                          ω -> begin
                              ω <= Δ && return 0
                              x = ω / Δ
                              rp = sqrt(2 * (x - 1))
                              # The second term regularizes the derivative
                              return (1.0 / rp + (3 / 8) * rp) / (2D)
                          end,
                          sing_integral
                         )
        ]
    )
end

raw"""
Return a `Keldysh.SingularDOS` object corresponding to the anomalous component of G(z), i.e.

    -\frac{1}{\pi} \Im[G_{12}(ϵ)].

# Parameters
- `D`: Half-bandwidth of the flat normal state DOS ρ(ϵ).
- `Δ`: Superconducting gap.
"""
function anomalous_dos(; D=5.0, Δ=1.0)
    @assert Δ > 0
    @assert D > Δ
    ωmax = sqrt(D^2 + Δ^2)
    xmax = ωmax / Δ
    sing_integral = (13 - xmax) / (12 * sqrt(2 * (1 + xmax)))

    kd.SingularDOS(-ωmax, ωmax,
        ω -> begin
            if abs(ω) <= Δ # ω in the superconducting gap
                return .0
            else
                x = ω / Δ
                r = sign(x) / sqrt(x^2 - 1)

                if x > 0
                    rp = sqrt(2 * (x - 1))
                    # The second term regularizes the derivative
                    r -= 1 / rp - (1 / 8) * rp
                else
                    rm = sqrt(-2 * (x + 1))
                    # The second term regularizes the derivative
                    r -= -(1 / rm - (1 / 8) * rm)
                end

                return r / (2D)
            end
        end,
        [
        kd.DOSSingularity(-Δ,
                          ω -> begin
                              ω >= -Δ && return 0
                              x = ω / Δ
                              x = ω / Δ
                              rm = sqrt(-2 * (x + 1))
                              # The second term regularizes the derivative
                              return -(1.0 / rm - (1 / 8) * rm) / (2D)
                          end,
                          -sing_integral
                         ),
        kd.DOSSingularity(Δ,
                          ω -> begin
                              ω <= Δ && return 0
                              x = ω / Δ
                              rp = sqrt(2 * (x - 1))
                              # The second term regularizes the derivative
                              return (1.0 / rp - (1 / 8) * rp) / (2D)
                          end,
                          sing_integral
                         )
        ]
    )
end

using Test

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "dos" begin
        dos_integrator = kd.GaussKronrodDOSIntegrator()

        # Normal component
        let Δ = 3.0, D = 6.0, dos = normal_dos(D=D, Δ=Δ)
            moments = [dos_integrator(ω -> ω^n, dos) for n = 0:6]
            moments_ref = [1.0,
                           0.0,
                           Δ^2 + D^2 / 3,
                           0,
                           Δ^4 + (2 / 3) * D^2 * Δ^2 + D^4 / 5,
                           0,
                           Δ^6 + Δ^4 * D^2 + (3 / 5) * Δ^2 * D^4 + D^6 / 7]
            @test isapprox(moments, moments_ref, atol=1e-10, rtol=1e-10)
        end

        # Anomalous component
        let Δ = 3.0, D = 6.0, dos = anomalous_dos(D=D, Δ=Δ)
            moments = [dos_integrator(ω -> ω^n, dos) for n = 0:6]
            moments_ref = [0.0,
                           Δ,
                           0.0,
                           Δ * (Δ^2 + D^2 / 3),
                           0.0,
                           Δ * (Δ^4 + (2 / 3) * Δ^2 * D^2 + D^4 / 5),
                           0.0]
            @test isapprox(moments, moments_ref, atol=1e-10, rtol=1e-10)
        end
    end
end
