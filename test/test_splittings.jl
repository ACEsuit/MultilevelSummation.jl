using Test
using MLSum
using StaticArrays
using StableRNGs

@testset "splittings: HardyC2Cubic" begin
    rng = StableRNG(0x5AA5)

    @testset "softening γ — value and slope match at R=1" begin
        ε = 1e-7
        γ_minus  = MLSum.gamma_softening(1.0 - ε)
        γ_plus   = MLSum.gamma_softening(1.0 + ε)
        # At R = 1: γ = 1 from both pieces (polynomial → 1; 1/R → 1)
        @test isapprox(γ_minus, 1.0; atol=1e-6)
        @test isapprox(γ_plus,  1.0; atol=1e-6)
        @test isapprox(γ_minus, γ_plus; atol=1e-6)
        # γ'(1⁻) = 1·(3/2 - 5/2) = -1; γ'(1⁺) = -1
        gp_minus = MLSum.gamma_softening_prime(1.0 - ε)
        gp_plus  = MLSum.gamma_softening_prime(1.0 + ε)
        @test isapprox(gp_minus, -1.0; atol=1e-6)
        @test isapprox(gp_plus,  -1.0; atol=1e-6)
        @test isapprox(gp_minus, gp_plus; atol=1e-6)
    end

    @testset "γ matches 1/R for R > 1" begin
        for R in (1.1, 2.0, 5.0, 17.3)
            @test MLSum.gamma_softening(R) ≈ 1 / R
            @test MLSum.gamma_softening_prime(R) ≈ -1 / R^2
        end
    end

    @testset "telescoping: k₀ + Σ kₗ + k_L = 1/|r|, D=$D, L=$L" for D in (1, 2, 3), L in (1, 2, 3, 4)
        a = 1.7
        s = HardyC2Cubic(a, L)
        for _ in 1:20
            r = SVector{D,Float64}((randn(rng, D) .* 0.5)...)
            sum(abs2, r) > 1e-6 || continue
            sr = sqrt(sum(abs2, r))
            k0 = short_range(s, r)
            klong = sum(long_range_level(s, l, r) for l in 1:(L-1); init=0.0)
            kL = top_level(s, r)
            @test isapprox(k0 + klong + kL, 1 / sr; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "k₀ compact support |r| ≤ a" begin
        a = 1.3
        s = HardyC2Cubic(a, 3)
        for D in (1, 2, 3), _ in 1:10
            # pick |r| > a guaranteed
            r = SVector{D,Float64}((randn(rng, D) .+ 5)...)
            @test abs(short_range(s, r)) < 1e-14
        end
    end

    @testset "gradient vs finite differences (per level)" begin
        a = 1.5
        L = 3
        s = HardyC2Cubic(a, L)
        for D in (1, 2, 3)
            for _ in 1:5
                # Choose r where everything is smooth: |r| < a, well away from |r|=a_l boundaries.
                r = SVector{D,Float64}((randn(rng, D) .* 0.3 .+ 0.1)...)
                sum(abs2, r) > 0.01 || continue

                fd_grad(f, r) = SVector{D,Float64}(ntuple(α -> begin
                    δ = 1e-6
                    e = SVector{D,Float64}(ntuple(β -> β == α ? δ : 0.0, D)...)
                    (f(r + e) - f(r - e)) / (2δ)
                end, D)...)

                @test isapprox(short_range_grad(s, r), fd_grad(r -> short_range(s, r), r);
                               atol=1e-5, rtol=1e-5)
                for l in 1:(L-1)
                    @test isapprox(long_range_level_grad(s, l, r),
                                   fd_grad(r -> long_range_level(s, l, r), r);
                                   atol=1e-5, rtol=1e-5)
                end
                @test isapprox(top_level_grad(s, r), fd_grad(r -> top_level(s, r), r);
                               atol=1e-5, rtol=1e-5)
            end
        end
    end

    @testset "Float32 precision" begin
        a = 1.5f0
        s = HardyC2Cubic(a, 3)
        r = SVector{3,Float32}(0.3f0, -0.2f0, 0.4f0)
        @test short_range(s, r) isa Float32
        @test long_range_level(s, 1, r) isa Float32
        @test top_level(s, r) isa Float32
        sr = sqrt(sum(abs2, r))
        k0 = short_range(s, r)
        k1 = long_range_level(s, 1, r)
        k2 = long_range_level(s, 2, r)
        kL = top_level(s, r)
        @test isapprox(k0 + k1 + k2 + kL, 1 / sr; atol=1e-5, rtol=1e-5)
    end

    @testset "flag: requires_neutralising_background" begin
        s = HardyC2Cubic(1.0, 2)
        @test requires_neutralising_background(s) == true
    end

    @testset "construction errors" begin
        @test_throws ArgumentError HardyC2Cubic(-1.0, 2)
        @test_throws ArgumentError HardyC2Cubic( 1.0, 0)
    end
end
