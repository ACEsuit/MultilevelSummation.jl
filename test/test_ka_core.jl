using Test
using MultilevelSummation
using MultilevelSummation.Reference: ewald_energy, naive_energy
using KernelAbstractions
using StaticArrays
using StableRNGs

# End-to-end paired comparison: msm_energy / msm_energy_forces with the
# legacy CPU path vs the KA-on-CPU path. Same inputs, same calculator,
# different `backend` kwarg — must agree to within roundoff.

@testset "MSM end-to-end (KA): open BC, Coulomb, D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xCA60E2 + D)
    N = 6
    positions = [SVector{D,Float64}((rand(rng, D) .* 2)...) for _ in 1:N]
    charges   = randn(rng, N) ./ sqrt(N)
    cell      = zero(SMatrix{D,D,Float64})
    periodic  = ntuple(_ -> false, D)

    a = 1.6
    h = 0.2
    L = 3
    calc = MSMCalculator(HardyC2Cubic(a, L), CubicC1(), h)

    U_cpu      = msm_energy(positions, charges, cell, periodic, calc)
    U_ka_cpu   = msm_energy(positions, charges, cell, periodic, calc; backend = CPU())
    @test isapprox(U_cpu, U_ka_cpu; atol=1e-10, rtol=1e-10)

    Uf_cpu, F_cpu     = msm_energy_forces(positions, charges, cell, periodic, calc)
    Uf_ka,  F_ka      = msm_energy_forces(positions, charges, cell, periodic, calc; backend = CPU())
    @test isapprox(Uf_cpu, Uf_ka; atol=1e-10, rtol=1e-10)
    @test isapprox(F_cpu, F_ka; atol=1e-10, rtol=1e-10)
end

@testset "MSM end-to-end (KA): fully periodic 3D, Coulomb vs Ewald" begin
    rng = StableRNG(0xCA60E3)

    h = 0.5
    L = 4
    cell_L = h * 8                   # 8 fine-grid points per axis → top is 1×1×1
    cell = SMatrix{3,3,Float64}(cell_L * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)

    N = 8
    positions = [SVector{3,Float64}((rand(rng, 3) .* cell_L)...) for _ in 1:N]
    charges   = randn(rng, N)
    charges  .-= sum(charges) / N    # neutralise

    a = 1.5
    calc = MSMCalculator(HardyC2Cubic(a, L), CubicC1(), h)

    U_cpu, F_cpu = msm_energy_forces(positions, charges, cell, periodic, calc)
    U_ka,  F_ka  = msm_energy_forces(positions, charges, cell, periodic, calc; backend = CPU())

    @test isapprox(U_cpu, U_ka; atol=1e-10, rtol=1e-10)
    @test isapprox(F_cpu, F_ka; atol=1e-10, rtol=1e-10)
end

@testset "MSM end-to-end (KA): mixed BC (1 periodic, 1 open), D=2" begin
    rng = StableRNG(0xCA60E4)
    D = 2
    L_box = 4.0
    N = 6
    positions = [SVector{D,Float64}((rand(rng, D) .* L_box)...) for _ in 1:N]
    charges   = randn(rng, N) ./ sqrt(N)
    cell      = SMatrix{D,D,Float64}(L_box * one(SMatrix{D,D,Float64}))
    periodic  = (true, false)

    a = 1.0
    h = 0.25
    L = 3
    calc = MSMCalculator(HardyC2Cubic(a, L), CubicC1(), h)

    U_cpu, F_cpu = msm_energy_forces(positions, charges, cell, periodic, calc)
    U_ka,  F_ka  = msm_energy_forces(positions, charges, cell, periodic, calc; backend = CPU())

    @test isapprox(U_cpu, U_ka; atol=1e-10, rtol=1e-10)
    @test isapprox(F_cpu, F_ka; atol=1e-10, rtol=1e-10)
end
