using Test
using MultilevelSummation
using MultilevelSummation: anterpolate_ka!
using KernelAbstractions
using StaticArrays
using StableRNGs

@testset "anterpolate (KA scatter): D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xCA40A2 + D)
    h = 0.5
    n = 12
    basis = CubicC1()

    for periodic_tuple in (ntuple(_ -> true, D), ntuple(_ -> false, D))
        g = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                        ntuple(_ -> 0.0, D), periodic_tuple)

        N = 16
        positions = [SVector{D,Float64}((rand(rng, D) .* (n * h * 0.9))...) for _ in 1:N]
        charges   = randn(rng, N)

        @testset "matches CPU, periodic=$periodic_tuple" begin
            qg_cpu = zeros(g.size...); anterpolate!(qg_cpu, positions, charges, g, basis)
            qg_ka  = zeros(g.size...); anterpolate_ka!(qg_ka, positions, charges, g, basis, CPU())
            @test isapprox(qg_ka, qg_cpu; atol=1e-12, rtol=1e-12)
        end
    end
end
