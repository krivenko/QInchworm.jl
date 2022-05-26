using Test, Random

using QInchworm

import QInchworm.diagrammatics

@testset "diagrammatics" begin

  top = diagrammatics.Topology([1 => 2, 3 => 4])
  @test isvalid(top)

  @test diagrammatics.isconnected(top, 0) == false
  @test diagrammatics.isconnected(top, 1) == false
  @test diagrammatics.isconnected(top, 2) == false
  @test diagrammatics.isconnected(top, 3) == true
  @test diagrammatics.isconnected(top, 4) == true

  top = diagrammatics.Topology([1 => 4, 2 => 3])
  @test diagrammatics.isconnected(top, 0) == false
  @test diagrammatics.isconnected(top, 1) == false
  @test diagrammatics.isconnected(top, 2) == true
  @test diagrammatics.isconnected(top, 3) == true
  @test diagrammatics.isconnected(top, 4) == true

  top = diagrammatics.Topology([1 => 3, 2 => 4])
  @test diagrammatics.isconnected(top, 0) == false
  @test diagrammatics.isconnected(top, 1) == true
  @test diagrammatics.isconnected(top, 2) == true
  @test diagrammatics.isconnected(top, 3) == true
  @test diagrammatics.isconnected(top, 4) == true

end
