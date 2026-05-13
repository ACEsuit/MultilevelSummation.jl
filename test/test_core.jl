using Test
using MultilevelSummation
using MultilevelSummation.Reference: ewald_energy, ewald_energy_forces
using StaticArrays
using StableRNGs

# NOTE on FD-vs-analytic comparisons:
#   For open BC, MSM rebuilds the grid hierarchy each call with origin tied
#   to the particle bounding box. Shifting a single particle that happens to
#   be at the bbox extremum also shifts the grid origin, so U is piecewise-
#   smooth in individual `r_i` and FD-vs-analytic doesn't match exactly.
#   For periodic BC the grid is cell-fixed; FD agrees with analytic forces
#   to ~1e-10 (verified). We therefore restrict the FD checks to periodic BC.

@testset "MSM end-to-end: open BC, Coulomb, D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xE2E0 + D)
    K = Coulomb()
    basis = CubicC1()

    @testset "MSM agrees with naive at moderate h/a" begin
        N = 5
        positions = [SVector{D,Float64}((rand(rng, D) .* 2)...) for _ in 1:N]
        charges   = randn(rng, N) ./ sqrt(N)
        cell      = zero(SMatrix{D,D,Float64})
        periodic  = ntuple(_ -> false, D)

        # Larger a → smaller splitting error.
        a = 4.0
        h = 0.2
        L = 3
        calc = MSMCalculator(HardyC2Cubic(a, L), basis, h)

        U_msm   = msm_energy(positions, charges, cell, periodic, calc)
        U_naive = naive_energy(positions, charges, cell, periodic, K)
        rel_err = abs(U_msm - U_naive) / max(abs(U_naive), 1e-10)
        # Realistic threshold — cubic basis + small grid gives a few % on
        # random configurations (configuration-dependent). The convergence-in-a
        # test below verifies the algorithm has the right scaling; this just
        # checks that nothing is grossly broken.
        @test rel_err < 0.20
    end

    @testset "Convergence in a (fixed h)" begin
        N = 5
        positions = [SVector{D,Float64}((rand(rng, D) .* 2)...) for _ in 1:N]
        charges   = randn(rng, N) ./ sqrt(N)
        cell      = zero(SMatrix{D,D,Float64})
        periodic  = ntuple(_ -> false, D)
        U_naive   = naive_energy(positions, charges, cell, periodic, K)

        h = 0.2
        L = 3
        errs = Float64[]
        for a in (0.8, 1.6, 3.2)
            calc = MSMCalculator(HardyC2Cubic(a, L), basis, h)
            U = msm_energy(positions, charges, cell, periodic, calc)
            push!(errs, abs(U - U_naive) / max(abs(U_naive), 1e-10))
        end
        @test errs[1] > errs[2] > errs[3]
    end

    @testset "Translation invariance (open, simultaneous shift of all)" begin
        N = 4
        positions = [SVector{D,Float64}((rand(rng, D) .* 1.0)...) for _ in 1:N]
        charges   = randn(rng, N) ./ sqrt(N)
        cell      = zero(SMatrix{D,D,Float64})
        periodic  = ntuple(_ -> false, D)

        a = 1.6
        h = 0.2
        L = 3
        calc = MSMCalculator(HardyC2Cubic(a, L), basis, h)

        U0 = msm_energy(positions, charges, cell, periodic, calc)
        # Translate ALL particles by the same vector — origin tracks, so U stays.
        shift = SVector{D,Float64}(ntuple(_ -> 5.7, D)...)
        positions_shifted = [r + shift for r in positions]
        U1 = msm_energy(positions_shifted, charges, cell, periodic, calc)
        @test isapprox(U0, U1; atol=1e-10, rtol=1e-8)
    end
end

@testset "MSM end-to-end: fully periodic 3D, Coulomb vs Ewald" begin
    rng = StableRNG(0xE3D3)

    h = 0.5
    L = 4
    cell_L = h * 8                   # 8 fine-grid points per axis → top is 1×1×1
    cell = SMatrix{3,3,Float64}(cell_L * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)

    N = 8
    positions = [SVector{3,Float64}((rand(rng, 3) .* cell_L)...) for _ in 1:N]
    charges   = randn(rng, N)
    charges  .-= sum(charges) / N

    basis = CubicC1()

    α     = 0.7
    R_cut = 10.0
    k_cut = 12.0
    U_ewald = ewald_energy(positions, charges, cell;
                            α=α, R_cut=R_cut, k_cut=k_cut)

    # As a grows the splitting error shrinks; check that MSM tracks Ewald.
    errs = Float64[]
    for a in (1.5, 2.0, 3.0)
        calc = MSMCalculator(HardyC2Cubic(a, L), basis, h)
        U_msm = msm_energy(positions, charges, cell, periodic, calc)
        @test isfinite(U_msm)
        push!(errs, abs(U_msm - U_ewald) / abs(U_ewald))
    end
    # Monotone decrease with growing a.
    @test errs[1] > errs[2] > errs[3]
    # And the largest-a value should be reasonably close.
    @test errs[end] < 0.05
end

@testset "MSM end-to-end: periodic FD gradient check" begin
    rng = StableRNG(0xE3DF)
    h = 0.5
    L = 4
    cell_L = h * 8
    cell = SMatrix{3,3,Float64}(cell_L * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    N = 5
    positions = [SVector{3,Float64}((rand(rng, 3) .* cell_L)...) for _ in 1:N]
    charges   = randn(rng, N)
    charges  .-= sum(charges) / N

    a = 2.0
    basis = CubicC1()
    calc = MSMCalculator(HardyC2Cubic(a, L), basis, h)
    U, F = msm_energy_forces(positions, charges, cell, periodic, calc)
    f_of(p) = msm_energy(p, charges, cell, periodic, calc)

    for i in 1:N, α in 1:3
        δ = 1e-5
        e = SVector{3,Float64}(ntuple(β -> β == α ? δ : 0.0, 3)...)
        p_plus  = copy(positions); p_plus[i]  = positions[i] + e
        p_minus = copy(positions); p_minus[i] = positions[i] - e
        fd = (f_of(p_plus) - f_of(p_minus)) / (2δ)
        @test isapprox(-F[i][α], fd; atol=1e-8, rtol=1e-6)
    end
end

@testset "MSM end-to-end: lattice translation invariance, periodic" begin
    rng = StableRNG(0xE314)
    h = 0.5
    L = 4
    cell_L = h * 8
    cell = SMatrix{3,3,Float64}(cell_L * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    N = 6
    positions = [SVector{3,Float64}((rand(rng, 3) .* cell_L)...) for _ in 1:N]
    charges   = randn(rng, N)
    charges  .-= sum(charges) / N

    a = 2.0
    basis = CubicC1()
    calc = MSMCalculator(HardyC2Cubic(a, L), basis, h)

    U0 = msm_energy(positions, charges, cell, periodic, calc)
    # Translate by a full lattice vector → exact invariance.
    positions_shifted = [SVector(r[1] + cell_L, r[2], r[3]) for r in positions]
    U1 = msm_energy(positions_shifted, charges, cell, periodic, calc)
    @test isapprox(U0, U1; atol=1e-10, rtol=1e-8)
end
