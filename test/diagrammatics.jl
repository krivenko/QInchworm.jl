using QInchworm.diagrammatics

@testset "diagrammatics" begin
  @test diagrammatics.double_factorial.(2:2:10) == [2, 8, 48, 384, 3840]
  @test diagrammatics.double_factorial.(1:2:10) == [1, 3, 15, 105, 945]

  for n = 1:7
    partitions = diagrammatics.pair_partitions(n)
    @test length(partitions) == diagrammatics.double_factorial(2n - 1)

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
