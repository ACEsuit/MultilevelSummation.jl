using Test
using MultilevelSummation.Tune: build_nacl, build_h2o
using StaticArrays
using Random
using LinearAlgebra: norm, ⋅

@testset "Tune.build_nacl" begin
    @testset "perfect lattice (σ = 0)" begin
        pos, q, cell, per = build_nacl(2)
        @test length(pos) == 8 * 2^3                       # 8 ions per conventional cell
        @test length(q)   == length(pos)
        @test sum(q) == 0                                   # exact neutrality, 32 +1 and 32 -1
        @test count(==(+1.0), q) == 32
        @test count(==(-1.0), q) == 32
        @test cell[1, 1] == 8.0 && cell[2, 2] == 8.0 && cell[3, 3] == 8.0
        @test per == (true, true, true)
        # Perfect lattice ⇒ deterministic positions independent of rng.
        pos2, _, _, _ = build_nacl(2; rng = MersenneTwister(123))
        @test pos == pos2
    end

    @testset "thermal sample (σ > 0) is reproducible from seed" begin
        pos_a, q_a, _, _ = build_nacl(2; σ = 0.1, rng = MersenneTwister(42))
        pos_b, q_b, _, _ = build_nacl(2; σ = 0.1, rng = MersenneTwister(42))
        @test pos_a == pos_b
        @test q_a   == q_b
        # σ = 0.1 should displace each ion by ~0.1 Å, not zero.
        pos0, _, _, _ = build_nacl(2)
        @test all(norm(pos_a[i] - pos0[i]) > 0 for i in eachindex(pos_a))
        @test all(norm(pos_a[i] - pos0[i]) < 1.0 for i in eachindex(pos_a))  # sanity: not huge
    end

    @testset "scales with n_super" begin
        for n in (1, 2, 3, 4)
            pos, q, cell, _ = build_nacl(n)
            @test length(pos) == 8 * n^3
            @test sum(q)      == 0
            @test cell[1, 1]  == 4.0 * n
        end
    end
end

@testset "Tune.build_h2o" begin
    @testset "basic shape, density, neutrality" begin
        box = 12.0
        pos, q, cell, per = build_h2o(box)
        n_mol = round(Int, 0.0334 * box^3)
        @test length(pos) == 3 * n_mol                      # O + 2 H per molecule
        @test length(q)   == length(pos)
        # Net charge ~ 0 (TIP3P: q_O = -0.834, q_H = +0.417; per-molecule sum = 0).
        @test abs(sum(q)) < 1e-10
        @test cell[1, 1] == box && cell[2, 2] == box && cell[3, 3] == box
        @test per == (true, true, true)
    end

    @testset "rigid molecular geometry" begin
        pos, _, _, _ = build_h2o(12.0)
        for m in 1:div(length(pos), 3)
            O  = pos[3*(m-1) + 1]
            H1 = pos[3*(m-1) + 2]
            H2 = pos[3*(m-1) + 3]
            @test isapprox(norm(O - H1), 0.9572; atol = 1e-9)
            @test isapprox(norm(O - H2), 0.9572; atol = 1e-9)
            # H-O-H angle = 104.52°. cos(θ) of the (H1-O), (H2-O) unit vectors.
            v1 = (H1 - O) / norm(H1 - O)
            v2 = (H2 - O) / norm(H2 - O)
            @test isapprox(acosd(clamp(v1 ⋅ v2, -1, 1)), 104.52; atol = 1e-6)
        end
    end

    @testset "Poisson-disk respects d_min" begin
        box = 16.0
        d_min = 2.7
        pos, _, _, _ = build_h2o(box; d_min = d_min)
        # Check pairwise min-image O-O distances. O atoms are at indices 1,4,7,...
        Oxs = [pos[3*(m-1) + 1] for m in 1:div(length(pos), 3)]
        for i in 1:length(Oxs), j in i+1:length(Oxs)
            dx = Oxs[j] - Oxs[i]
            dx = SVector{3,Float64}(
                dx[1] - box * round(dx[1] / box),
                dx[2] - box * round(dx[2] / box),
                dx[3] - box * round(dx[3] / box),
            )
            @test norm(dx) ≥ d_min - 1e-12
        end
    end

    @testset "reproducible from seed" begin
        pos_a, _, _, _ = build_h2o(12.0; rng = MersenneTwister(99))
        pos_b, _, _, _ = build_h2o(12.0; rng = MersenneTwister(99))
        @test pos_a == pos_b
    end
end
