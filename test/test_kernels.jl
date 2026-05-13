using Test
using MultilevelSummation
using StaticArrays
using StableRNGs

@testset "kernels" begin
    rng = StableRNG(0xC0FFEE)

    @testset "InversePower{$N}" for N in (1, 2, 4, 6)
        K = InversePower{N,Float64}()
        for D in (1, 2, 3), _ in 1:10
            r = SVector{D,Float64}(randn(rng, D)...)
            s = sqrt(sum(abs2, r))
            # value
            @test K(r) ≈ 1 / s^N
            # gradient by finite differences
            g = MultilevelSummation.grad(K, r)
            for α in 1:D
                δ = 1e-6
                e = SVector{D,Float64}(ntuple(β -> β == α ? δ : 0.0, D)...)
                fd = (K(r + e) - K(r - e)) / (2δ)
                @test isapprox(g[α], fd; atol=1e-6, rtol=1e-6)
            end
        end
    end

    @testset "InversePower{$N,Float32}" for N in (1, 2, 6)
        K = InversePower{N,Float32}()
        r = SVector{3,Float32}(1.5f0, -0.3f0, 0.7f0)
        @test K(r) isa Float32
        @test MultilevelSummation.grad(K, r) isa SVector{3,Float32}
    end

    @testset "Coulomb alias" begin
        @test Coulomb() === InversePower{1,Float64}()
        @test Coulomb{Float32}() isa InversePower{1,Float32}
    end

    @testset "RationalDecay{$N}" for N in (2, 4, 6)
        r₀ = 1.7
        K = RationalDecay{N}(r₀)
        for D in (1, 2, 3), _ in 1:10
            r = SVector{D,Float64}(randn(rng, D)...)
            s = sqrt(sum(abs2, r))
            @test K(r) ≈ 1 / (1 + (s / r₀)^N)
            # gradient by FD (away from origin; safe for N >= 2)
            g = MultilevelSummation.grad(K, r)
            for α in 1:D
                δ = 1e-6
                e = SVector{D,Float64}(ntuple(β -> β == α ? δ : 0.0, D)...)
                fd = (K(r + e) - K(r - e)) / (2δ)
                @test isapprox(g[α], fd; atol=1e-6, rtol=1e-6)
            end
        end
    end

    @testset "RationalDecay smooth at origin" begin
        K = RationalDecay{4}(1.0)
        @test K(SVector(0.0, 0.0, 0.0)) ≈ 1.0
    end
end
