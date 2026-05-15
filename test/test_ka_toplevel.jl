using Test
using MultilevelSummation
using MultilevelSummation: top_level_ka!
using KernelAbstractions
using StaticArrays
using StableRNGs

@testset "top_level (KA): D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xCA20FE + D)
    h = 0.5
    n = 4      # small top grid
    L = 3
    a = 1.0
    splitting = HardyC2Cubic(a, L)

    for periodic_tuple in (ntuple(_ -> true, D), ntuple(_ -> false, D))
        g = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                        ntuple(_ -> 0.0, D), periodic_tuple)
        q = randn(rng, g.size...)
        e_cpu = zeros(g.size...);  top_level!(e_cpu, q, g, splitting)
        e_ka  = zeros(g.size...);  top_level_ka!(e_ka, q, g, splitting, CPU())
        @test isapprox(e_ka, e_cpu; atol=1e-12, rtol=1e-12)
    end
end
