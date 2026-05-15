using Test
using MultilevelSummation
using MultilevelSummation: interpolate_ka!, interpolate_grad_ka!
using KernelAbstractions
using StaticArrays
using StableRNGs

@testset "interpolate / interpolate_grad (KA): D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xCA30A1 + D)
    h = 0.5
    n = 12
    basis = CubicC1()

    for periodic_tuple in (ntuple(_ -> true, D), ntuple(_ -> false, D))
        g = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                        ntuple(_ -> 0.0, D), periodic_tuple)
        grid_vals = randn(rng, g.size...)

        N = 8
        positions = [SVector{D,Float64}((rand(rng, D) .* (n * h * 0.9))...) for _ in 1:N]

        @testset "interpolate, periodic=$periodic_tuple" begin
            pot_cpu = zeros(N); interpolate!(pot_cpu, positions, grid_vals, g, basis)
            pot_ka  = zeros(N); interpolate_ka!(pot_ka, positions, grid_vals, g, basis, CPU())
            @test isapprox(pot_ka, pot_cpu; atol=1e-12, rtol=1e-12)
        end

        @testset "interpolate_grad, periodic=$periodic_tuple" begin
            g_cpu = zeros(SVector{D,Float64}, N); interpolate_grad!(g_cpu, positions, grid_vals, g, basis)
            g_ka  = zeros(SVector{D,Float64}, N); interpolate_grad_ka!(g_ka, positions, grid_vals, g, basis, CPU())
            @test isapprox(g_cpu, g_ka; atol=1e-12, rtol=1e-12)
        end
    end
end
