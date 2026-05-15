using Test
using MultilevelSummation
using MultilevelSummation: restrict_ka!, prolong_ka!
using KernelAbstractions
using StaticArrays
using StableRNGs

@testset "restrict / prolong (KA): D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xCA1F60 + D)
    h = 0.5
    n = 16     # divisible by 2
    basis = CubicC1()

    for periodic_tuple in (ntuple(_ -> true, D), ntuple(_ -> false, D))
        src_grid = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                               ntuple(_ -> 0.0, D), periodic_tuple)
        dst_grid = coarser_grid(src_grid, 2)

        @testset "restrict, periodic=$periodic_tuple" begin
            src = randn(rng, src_grid.size...)
            dst_cpu = zeros(dst_grid.size...); restrict!(dst_cpu, src, dst_grid, src_grid, basis)
            dst_ka  = zeros(dst_grid.size...); restrict_ka!(dst_ka, src, dst_grid, src_grid, basis, CPU())
            @test isapprox(dst_ka, dst_cpu; atol=1e-12, rtol=1e-12)
        end

        @testset "prolong, periodic=$periodic_tuple" begin
            src = randn(rng, dst_grid.size...)
            # prolong target is the finer grid
            out_cpu = zeros(src_grid.size...); prolong!(out_cpu, src, src_grid, dst_grid, basis)
            out_ka  = zeros(src_grid.size...); prolong_ka!(out_ka, src, src_grid, dst_grid, basis, CPU())
            @test isapprox(out_ka, out_cpu; atol=1e-12, rtol=1e-12)
        end
    end
end
