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
# Author: Joseph Kleinhenz

using Test
using Combinatorics: doublefactorial

using QInchworm.diagrammatics

@testset "diagrammatics" begin
  for n = 1:7
    partitions = diagrammatics.pair_partitions(n)
    @test length(partitions) == doublefactorial(2n - 1)

    for p in partitions
      # test if we have generated valid partition
      @test sort!(collect(Iterators.flatten(p))) == 1:(2n)
    end
  end

  for n = 1:7
    topologies = diagrammatics.generate_topologies(n)
    for top in topologies
      @test isvalid(top)
    end
  end

  for n = 1:7
    topologies = diagrammatics.generate_topologies(n)
    for top in topologies
      for k in 1:n
        @test (-1)^k == (-1)^(diagrammatics.count_doubly_k_connected(top.pairs, k))
      end
    end
  end

  top = diagrammatics.Topology([1 => 2, 3 => 4])
  @test isvalid(top)

  @test diagrammatics.is_k_connected(top, 0) == false
  @test diagrammatics.is_k_connected(top, 1) == false
  @test diagrammatics.is_k_connected(top, 2) == false
  @test diagrammatics.is_k_connected(top, 3) == true
  @test diagrammatics.is_k_connected(top, 4) == true

  top = diagrammatics.Topology([1 => 4, 2 => 3])
  @test isvalid(top)
  @test diagrammatics.is_k_connected(top, 0) == false
  @test diagrammatics.is_k_connected(top, 1) == false
  @test diagrammatics.is_k_connected(top, 2) == true
  @test diagrammatics.is_k_connected(top, 3) == true
  @test diagrammatics.is_k_connected(top, 4) == true

  top = diagrammatics.Topology([1 => 3, 2 => 4])
  @test isvalid(top)
  @test diagrammatics.is_k_connected(top, 0) == false
  @test diagrammatics.is_k_connected(top, 1) == true
  @test diagrammatics.is_k_connected(top, 2) == true
  @test diagrammatics.is_k_connected(top, 3) == true
  @test diagrammatics.is_k_connected(top, 4) == true
end
