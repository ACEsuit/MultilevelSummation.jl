using Test
using MultilevelSummation
using StaticArrays
using StableRNGs

# Build a periodic fine grid and its 2× coarsened companion.
function _two_level_grids(::Val{D}, h::T, n::Int) where {D,T}
    @assert iseven(n) "n must be even for 2× coarsening"
    fine   = UniformGrid(ntuple(_ -> h,     D),  ntuple(_ -> n,      D),
                         ntuple(_ -> zero(T), D), ntuple(_ -> true, D))
    coarse = coarser_grid(fine, 2)
    return fine, coarse
end

@testset "restrict / prolong: D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xB011D + D)
    h, n = 0.5, 8
    fine, coarse = _two_level_grids(Val(D), h, n)
    b = CubicC1()

    @testset "coarser_grid: doubles spacing, halves extent" begin
        @test all(coarse.spacing .≈ 2 .* fine.spacing)
        @test coarse.size == ntuple(_ -> 4, D)
        @test coarse.periodic == fine.periodic
        @test coarse.origin == fine.origin
    end

    @testset "transpose: ⟨c, R f⟩ = ⟨P c, f⟩" begin
        f = randn(rng, fine.size...)
        c = randn(rng, coarse.size...)
        # Rf
        Rf = zeros(coarse.size...)
        restrict!(Rf, f, coarse, fine, b)
        lhs = sum(c .* Rf)
        # Pc
        Pc = zeros(fine.size...)
        prolong!(Pc, c, fine, coarse, b)
        rhs = sum(Pc .* f)
        @test isapprox(lhs, rhs; atol=1e-12, rtol=1e-12)
    end

    @testset "restrict preserves total weight (periodic, partition of unity)" begin
        f = randn(rng, fine.size...)
        Rf = zeros(coarse.size...)
        restrict!(Rf, f, coarse, fine, b)
        @test isapprox(sum(Rf), sum(f); atol=1e-10, rtol=1e-10)
    end

    @testset "prolong of a constant returns the constant" begin
        c = fill(1.7, coarse.size...)
        Pc = zeros(fine.size...)
        prolong!(Pc, c, fine, coarse, b)
        for v in Pc
            @test isapprox(v, 1.7; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "prolong reproduces polynomials degree ≤ 2 (open BC)" begin
        # On an open grid, the prolongation of p sampled at coarse points
        # should equal p sampled at fine points (away from boundaries).
        # Build a (large) open grid pair so interior strips are clean.
        n_o = 16
        fine_o   = UniformGrid(ntuple(_ -> h,    D), ntuple(_ -> n_o,    D),
                               ntuple(_ -> 0.0, D), ntuple(_ -> false, D))
        coarse_o = coarser_grid(fine_o, 2)
        # Polynomial p(r) = a + Σ b_α r_α + Σ c_α r_α²
        a0 = rand(rng); bs = rand(rng, D); cs = rand(rng, D) .* 0.3
        eval_p = r -> begin
            v = a0
            @inbounds for α in 1:D
                v += bs[α] * r[α] + cs[α] * r[α]^2
            end
            v
        end
        c = zeros(Float64, coarse_o.size...)
        @inbounds for I in CartesianIndices(c)
            r = SVector{D,Float64}(ntuple(α -> (I[α] - 1) * coarse_o.spacing[α], D))
            c[I] = eval_p(r)
        end
        Pc = zeros(Float64, fine_o.size...)
        prolong!(Pc, c, fine_o, coarse_o, b)

        # Compare interior fine points (avoid boundary strip of support radius)
        s = support_radius(b)
        for I in CartesianIndices(Pc)
            in_interior = all(α -> 2s ≤ I[α] - 1 ≤ n_o - 1 - 2s, 1:D)
            in_interior || continue
            r = SVector{D,Float64}(ntuple(α -> (I[α] - 1) * fine_o.spacing[α], D))
            @test isapprox(Pc[I], eval_p(r); atol=1e-10, rtol=1e-10)
        end
    end

    @testset "prolong-then-restrict damps high-frequency content" begin
        # Construct a fine-grid field that's highly oscillatory (alternating ±1).
        # After R then P (and back), it should be heavily damped, because R∘P kills
        # frequencies above the coarse Nyquist.
        f_hi = zeros(fine.size...)
        @inbounds for I in CartesianIndices(f_hi)
            f_hi[I] = (-1.0)^(sum(I.I))
        end
        Rf = zeros(coarse.size...); restrict!(Rf, f_hi, coarse, fine, b)
        Pf = zeros(fine.size...);  prolong!(Pf, Rf, fine, coarse, b)
        @test maximum(abs, Pf) < 0.5  # high-frequency content damped well below 1
    end
end
