using Test

import Keldysh; kd = Keldysh

import QInchworm.utility: get_ref

@testset "get_ref()" begin
    tmax = 1.
    β = 5.

    contour = kd.twist(kd.FullContour(tmax=tmax, β=β));

    for ref in [0.0, 0.5, 2.0, 5.0, 5.5, 6.5]
        @test get_ref(contour, contour(ref)) == ref
    end
end
