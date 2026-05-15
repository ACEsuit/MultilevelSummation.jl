using Test
using MultilevelSummation
using MultilevelSummation: grid_cutoff_ka!
using KernelAbstractions
using StaticArrays
using StableRNGs

@testset "grid cutoff (KA): D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xCA0F61 + D)
    h, n = 0.5, 12
    g = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                    ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
    L = 3
    a = h
    splitting = HardyC2Cubic(a, L)

    @testset "matches CPU implementation, level=$l" for l in 1:(L-1)
        q   = randn(rng, g.size...)
        e_cpu = zeros(g.size...);  grid_cutoff!(e_cpu, q, g, splitting, l)
        e_ka  = zeros(g.size...);  grid_cutoff_ka!(e_ka, q, g, splitting, l, CPU())
        @test isapprox(e_ka, e_cpu; atol=1e-12, rtol=1e-12)
    end

    @testset "open BC: matches CPU, level=1" begin
        g_open = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                             ntuple(_ -> 0.0, D), ntuple(_ -> false, D))
        q = randn(rng, g_open.size...)
        e_cpu = zeros(g_open.size...);  grid_cutoff!(e_cpu, q, g_open, splitting, 1)
        e_ka  = zeros(g_open.size...);  grid_cutoff_ka!(e_ka, q, g_open, splitting, 1, CPU())
        @test isapprox(e_ka, e_cpu; atol=1e-12, rtol=1e-12)
    end
end
