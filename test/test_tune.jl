using Test
using MultilevelSummation
using MultilevelSummation.Tune
using MultilevelSummation.Tune: SweepResult
using MultilevelSummation.Reference: ewald_energy
using StaticArrays
using StableRNGs

# Small periodic Coulomb system that fits MSM's n_grid divisibility constraints
# at multiple (h, L) combos for h ∈ {0.5, 1.0, 2.0}.
function _periodic_neutral_system(N::Int, h::Float64, n_fine::Int; seed = 0xCE11)
    rng = StableRNG(seed)
    box = h * n_fine
    cell = SMatrix{3,3,Float64}(box * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    positions = [SVector{3,Float64}((rand(rng, 3) .* box)...) for _ in 1:N]
    charges   = randn(rng, N)
    charges  .-= sum(charges) / N
    return positions, charges, cell, periodic
end

@testset "Tune" begin
    positions, charges, cell, periodic = _periodic_neutral_system(8, 0.5, 16)

    @testset "ewald_reference matches direct ewald_energy" begin
        U_tune = Tune.ewald_reference(positions, charges, cell; tol = 1e-9)
        # Re-derive the params the same way the Tune wrapper does, call the
        # underlying reference directly, and compare bit-for-bit.
        box   = cell[1, 1]
        αR    = sqrt(-log(1e-9))
        R_cut = min(box / 2 - 0.5, 14.0)
        α     = αR / R_cut
        k_cut = 2 * α * αR
        U_direct = ewald_energy(positions, charges, cell;
                                α = α, R_cut = R_cut, k_cut = k_cut)
        @test U_tune == U_direct
    end

    @testset "sweep produces SweepResult rows consistent with msm_energy" begin
        U_ref = Tune.ewald_reference(positions, charges, cell; tol = 1e-9)

        results = Tune.sweep(positions, charges, cell, periodic;
                              reference   = U_ref,
                              h_values    = (0.5, 1.0),
                              a_values    = (1.0, 2.0),
                              L_strategy  = :all)

        @test !isempty(results)
        @test eltype(results) == SweepResult{Float64}

        # Each row's energy + rel_err must match what we'd compute directly.
        for r in results
            calc = MSMCalculator(HardyC2Cubic(r.a, r.L), CubicC1(), r.h)
            U_check = msm_energy(positions, charges, cell, periodic, calc)
            @test isapprox(U_check, r.energy; atol = 0, rtol = 1e-12)
            @test isapprox(r.rel_err, abs(U_check - U_ref) / abs(U_ref);
                           atol = 0, rtol = 1e-12)
        end
    end

    @testset "sweep L_strategy = :max returns one L per h" begin
        U_ref = Tune.ewald_reference(positions, charges, cell; tol = 1e-9)
        max_results = Tune.sweep(positions, charges, cell, periodic;
                                  reference  = U_ref,
                                  h_values   = (0.5, 1.0),
                                  a_values   = (1.0,),
                                  L_strategy = :max)
        # For h ∈ {0.5, 1.0} and a = 1.0 the (n_grid, L_max) pairs are
        # well-defined and unique → exactly one row per h.
        @test length(max_results) == 2
        # And both rows pick the maximum valid L for their grid.
        for r in max_results
            ng = round(Int, cell[1, 1] / r.h)
            L = 1
            while rem(ng, 2^L) == 0 && L < 6
                L += 1
            end
            @test r.L == L
            @test r.n_grid == ng
        end
    end

    @testset "pareto_front" begin
        # Hand-built results: 1 is dominated by 2; 3 trades accuracy for speed.
        # Expected front (sorted by ascending rel_err): [2, 3].
        rs = [
            SweepResult{Float64}(0.5, 1.0, 3,  8, 0.0, 1e-3, 0.20),    # 1, dominated by 2
            SweepResult{Float64}(0.5, 2.0, 3,  8, 0.0, 1e-4, 0.15),    # 2
            SweepResult{Float64}(1.0, 1.0, 2,  4, 0.0, 1e-2, 0.05),    # 3
        ]
        front = Tune.pareto_front(rs)
        @test length(front) == 2
        @test front[1].rel_err < front[2].rel_err
        @test front[1] === rs[2]
        @test front[2] === rs[3]
    end

    @testset "recommend" begin
        rs = [
            SweepResult{Float64}(0.5, 2.0, 3,  8, 0.0, 1e-4, 0.15),
            SweepResult{Float64}(1.0, 1.0, 2,  4, 0.0, 1e-2, 0.05),
            SweepResult{Float64}(2.0, 4.0, 2,  2, 0.0, 5e-1, 0.01),
        ]
        # Strictest threshold — picks the slowest accurate one.
        @test Tune.recommend(rs; max_rel_err = 1e-4) === rs[1]
        # Looser — picks the faster one that meets it.
        @test Tune.recommend(rs; max_rel_err = 1e-1) === rs[2]
        # Below everyone — should throw.
        @test_throws ArgumentError Tune.recommend(rs; max_rel_err = 1e-10)
    end

    @testset "sweep rejects unknown L_strategy" begin
        U_ref = Tune.ewald_reference(positions, charges, cell; tol = 1e-9)
        @test_throws ArgumentError Tune.sweep(positions, charges, cell, periodic;
                                               reference  = U_ref,
                                               L_strategy = :nope)
    end
end
