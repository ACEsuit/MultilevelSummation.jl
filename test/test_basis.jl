using Test
using MLSum
using StableRNGs

@testset "basis" begin
    Φ = CubicC1()

    @testset "support and zeros" begin
        @test eval_phi(Φ, 2.0)  == 0      # boundary value: -(1/2)(1)(0)^2 = 0
        @test eval_phi(Φ, 2.5)  == 0
        @test eval_phi(Φ, -3.0) == 0
        @test support_radius(Φ) == 2
    end

    @testset "Φ(0) = 1 (interpolant at the node)" begin
        @test eval_phi(Φ, 0.0) ≈ 1.0
    end

    @testset "C¹ continuity across breakpoints" begin
        ε = 1e-8
        for ξ₀ in (-2.0, -1.0, 1.0, 2.0)
            @test isapprox(eval_phi(Φ, ξ₀ - ε), eval_phi(Φ, ξ₀ + ε); atol=1e-7)
            @test isapprox(eval_phi_prime(Φ, ξ₀ - ε),
                           eval_phi_prime(Φ, ξ₀ + ε); atol=1e-6)
        end
    end

    @testset "derivative vs finite difference" begin
        rng = StableRNG(0xBA51C)
        for _ in 1:50
            # Sample only inside one of the smooth pieces (avoid the breakpoints).
            ξ = 1.8 * (rand(rng) * 2 - 1)             # in (-1.8, 1.8)
            if abs(ξ) < 0.95 || abs(ξ) > 1.05         # avoid the breakpoint at ±1
                δ = 1e-7
                fd = (eval_phi(Φ, ξ + δ) - eval_phi(Φ, ξ - δ)) / (2δ)
                @test isapprox(eval_phi_prime(Φ, ξ), fd; atol=1e-6, rtol=1e-6)
            end
        end
    end

    @testset "partition of unity on the grid" begin
        # For cubic C¹ basis, Σ_m Φ(ξ - m) = 1 holds for ξ in the support
        # interior away from boundary effects. Test at random ξ summing over
        # integer offsets within the support window.
        rng = StableRNG(0xDADA)
        for _ in 1:50
            ξ = rand(rng) * 6 - 3                     # in (-3, 3)
            # Φ has support [-2,2], so contributing m are floor(ξ-2)..ceil(ξ+2)
            ms = floor(Int, ξ - 2):ceil(Int, ξ + 2)
            s = sum(eval_phi(Φ, ξ - m) for m in ms)
            @test isapprox(s, 1.0; atol=1e-12)
        end
    end

    @testset "polynomial reproduction up to degree 2 (exact)" begin
        # The paper's C¹ cubic basis is a linear blend of two quadratic
        # interpolants, so it exactly reproduces polynomials up to degree 2
        # and approximates cubics with O(h³) error. See paper §2.2.
        rng = StableRNG(0xFEED)
        polys = (ξ -> 1.0, ξ -> ξ, ξ -> ξ^2, ξ -> 1.7 - 2.3ξ + 0.8ξ^2)
        for p in polys, _ in 1:20
            ξ = rand(rng) * 4 - 2
            ms = floor(Int, ξ - 3):ceil(Int, ξ + 3)
            recon = sum(p(float(m)) * eval_phi(Φ, ξ - m) for m in ms)
            @test isapprox(recon, p(ξ); atol=1e-10, rtol=1e-10)
        end
    end

    @testset "cubic convergence O(h³) for x³ in max-norm" begin
        # Catmull-Rom cubic only exactly reproduces quadratics. For x^3 the
        # pointwise error oscillates with the fractional position ν ∈ [0,1].
        # The max-norm error over a full ν-period (taken here as 64 samples)
        # scales as h³.
        p = x -> x^3
        max_errs = Float64[]
        for h in (0.5, 0.25, 0.125, 0.0625)
            local_max = 0.0
            for ν in range(0.05, 0.95, length=64)
                x = (4 + ν) * h          # fix the local cell, sweep ν
                ξ = x / h
                ms = floor(Int, ξ - 3):ceil(Int, ξ + 3)
                recon = sum(p(m * h) * eval_phi(Φ, ξ - m) for m in ms)
                local_max = max(local_max, abs(recon - p(x)))
            end
            push!(max_errs, local_max)
        end
        # Each halving of h must shrink the max error by ~8×.
        for k in 2:length(max_errs)
            ratio = max_errs[k-1] / max_errs[k]
            @test 7.0 ≤ ratio ≤ 9.0
        end
    end
end
