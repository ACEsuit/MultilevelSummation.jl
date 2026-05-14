using Test
using MultilevelSummation
using AtomsBase
using AtomsCalculators
using StaticArrays
using StableRNGs
using Unitful

# Build a periodic charge system for testing.
function _periodic_charge_system(rng, L::Float64, N::Int)
    L_q = L * u"Å"
    cell_vec = (SVector(L_q, 0u"Å", 0u"Å"),
                SVector(0u"Å", L_q, 0u"Å"),
                SVector(0u"Å", 0u"Å", L_q))
    qs = randn(rng, N)
    qs .-= sum(qs) / N
    atoms = [Atom(:Ar,
                  SVector((rand(rng) * L) * u"Å",
                          (rand(rng) * L) * u"Å",
                          (rand(rng) * L) * u"Å"),
                  charge=qs[i])
             for i in 1:N]
    return periodic_system(atoms, cell_vec)
end

@testset "AtomsCalculator wrapper" begin
    rng = StableRNG(0xCA1C)
    L  = 4.0
    N  = 6
    sys = _periodic_charge_system(rng, L, N)

    basis = CubicC1()
    h = 0.5
    a = 2.0
    nlev = 4                   # L_levels=4 ⇒ top is 1×1×1 for an 8-point fine grid
    calc = MSMCalculator(HardyC2Cubic(a, nlev), basis, h)

    @testset "potential_energy matches low-level core" begin
        # Reproduce the same call on the unit-free core directly.
        D = 3
        positions = [SVector{D,Float64}(ntuple(α -> ustrip(position(sys, i)[α]), D))
                     for i in 1:N]
        charges   = [Float64(sys[i, :charge]) for i in 1:N]
        cell      = SMatrix{D,D,Float64}(reduce(hcat,
                    SVector{D,Float64}(ntuple(α -> ustrip(cell_vectors(sys)[β][α]), D))
                    for β in 1:D))
        U_core = msm_energy(positions, charges, cell, NTuple{D,Bool}(periodicity(sys)), calc)
        U_atoms = AtomsCalculators.potential_energy(sys, calc)
        # Relaxed from `==` to `isapprox` so threaded reduction order doesn't
        # break the assertion. Same inputs through the same code path should
        # still be bit-identical, but defensive against scheduler variation.
        @test isapprox(U_core, U_atoms; rtol = 1e-12)
    end

    @testset "forces and energy_forces consistency" begin
        F   = AtomsCalculators.forces(sys, calc)
        ef  = AtomsCalculators.energy_forces(sys, calc)
        # Relaxed from `==` to `isapprox` (see note above).
        @test all(isapprox.(ef.forces, F; rtol = 1e-12))
        # Energy obtained via energy_forces matches potential_energy.
        @test ef.energy ≈ AtomsCalculators.potential_energy(sys, calc)
    end

    @testset "forces! in-place" begin
        F = AtomsCalculators.forces(sys, calc)
        F2 = fill(zero(SVector{3,Float64}), N)
        AtomsCalculators.forces!(F2, sys, calc)
        # Relaxed from `==` to `isapprox` (see note above).
        @test all(isapprox.(F2, F; rtol = 1e-12))
    end

    @testset "FD check through AtomsCalculator API (periodic, smooth)" begin
        # Build an explicit array of positions, shift particle 1, build new system,
        # compute energy. Compare FD vs the analytic force.
        F = AtomsCalculators.forces(sys, calc)
        cell_vec = cell_vectors(sys)
        original_pos = [position(sys, i) for i in 1:N]
        charges_raw = [sys[i, :charge] for i in 1:N]

        for i in 1:N, α in 1:3
            δ_val = 1e-5u"Å"
            for sgn in (+1, -1)
                shifted = copy(original_pos)
                shifted[i] = SVector(ntuple(β -> shifted[i][β] + (β == α ? sgn * δ_val : 0u"Å"), 3))
                atoms = [Atom(:Ar, shifted[k], charge=charges_raw[k]) for k in 1:N]
                sys_shifted = periodic_system(atoms, cell_vec)
                U = AtomsCalculators.potential_energy(sys_shifted, calc)
                if sgn == +1
                    global U_p = U
                else
                    global U_m = U
                end
            end
            fd = (U_p - U_m) / (2 * 1e-5)
            @test isapprox(-F[i][α], fd; atol=1e-5, rtol=1e-5)
        end
    end

    @testset "custom charge_property name" begin
        # Build a system with charges under :q instead of :charge.
        atoms = [Atom(:Ar, position(sys, i); q = sys[i, :charge]) for i in 1:N]
        sys_q = periodic_system(atoms, cell_vectors(sys))
        calc_q = MSMCalculator(HardyC2Cubic(a, nlev), basis, h;
                                  charge_property = :q)
        U_q = AtomsCalculators.potential_energy(sys_q, calc_q)
        U_ref = AtomsCalculators.potential_energy(sys, calc)
        @test U_q ≈ U_ref
    end
end
