using Test
using MultilevelSummation
using MultilevelSummation.Tune: build_nacl, build_h2o
using KernelAbstractions
using StaticArrays

# Tolerance choice: GPU and CPU reductions disagree by a few ULPs from
# different summation orders. For Float64 on ≤ 1 000-atom systems,
# `rtol = 1e-5` is conservative (typical disagreement is 1e-13–1e-12);
# for Float32, `1e-2` is the right scale. We are testing correctness,
# not bit-identity.
const RTOL_F64 = 1e-5
const RTOL_F32 = 1e-2

"""
    run_equivalence_tests(backend, make_gpu_array)

Compare three computation paths on small NaCl and H2O fixtures:
1. Legacy CPU (`msm_energy_forces(positions, charges, ...)`).
2. Explicit-backend kwarg KA path (forces `_msm_compute_ka` via the
   kwarg branch of `_resolve_backend`).
3. Array-type-dispatched KA path (positions / charges moved to the GPU
   array type, no `backend` kwarg — forces `_resolve_backend` to read
   the backend off the array).

All three must agree to within `RTOL_F64` for Float64 fixtures. A
Float32 spot-check on NaCl ensures the type-parameterised path runs on
the GPU as well.
"""
function run_equivalence_tests(backend, make_gpu_array)
    @testset "NaCl (n_super=2)" begin
        positions, charges, cell, periodic = build_nacl(2)
        h = 1.0
        L = 3
        a = 2.0
        calc = MSMCalculator(HardyC2Cubic(a, L), CubicC1(), h)

        U_cpu, F_cpu       = msm_energy_forces(positions, charges, cell, periodic, calc)
        U_kw,  F_kw        = msm_energy_forces(positions, charges, cell, periodic, calc;
                                               backend = backend)
        positions_gpu      = make_gpu_array(positions)
        charges_gpu        = make_gpu_array(charges)
        U_at,  F_at        = msm_energy_forces(positions_gpu, charges_gpu, cell, periodic, calc)

        @test isapprox(U_kw, U_cpu; rtol = RTOL_F64)
        @test isapprox(U_at, U_cpu; rtol = RTOL_F64)
        @test isapprox(Array(F_kw), F_cpu; rtol = RTOL_F64)
        @test isapprox(Array(F_at), F_cpu; rtol = RTOL_F64)
    end

    @testset "H2O box (8 Å)" begin
        positions, charges, cell, periodic = build_h2o(8.0)
        h = 0.5
        L = 3
        a = 1.5
        calc = MSMCalculator(HardyC2Cubic(a, L), CubicC1(), h)

        U_cpu, F_cpu  = msm_energy_forces(positions, charges, cell, periodic, calc)
        U_kw,  F_kw   = msm_energy_forces(positions, charges, cell, periodic, calc;
                                          backend = backend)
        positions_gpu = make_gpu_array(positions)
        charges_gpu   = make_gpu_array(charges)
        U_at,  F_at   = msm_energy_forces(positions_gpu, charges_gpu, cell, periodic, calc)

        @test isapprox(U_kw, U_cpu; rtol = RTOL_F64)
        @test isapprox(U_at, U_cpu; rtol = RTOL_F64)
        @test isapprox(Array(F_kw), F_cpu; rtol = RTOL_F64)
        @test isapprox(Array(F_at), F_cpu; rtol = RTOL_F64)
    end

    @testset "NaCl (n_super=2) — Float32 spot-check" begin
        positions64, charges64, cell64, periodic = build_nacl(2)
        positions = [SVector{3,Float32}(p) for p in positions64]
        charges   = Float32.(charges64)
        cell      = SMatrix{3,3,Float32}(cell64)
        h, L, a   = 1.0f0, 3, 2.0f0
        calc      = MSMCalculator(HardyC2Cubic(a, L), CubicC1{Float32}(), h)

        U_cpu, F_cpu  = msm_energy_forces(positions, charges, cell, periodic, calc)
        positions_gpu = make_gpu_array(positions)
        charges_gpu   = make_gpu_array(charges)
        U_at,  F_at   = msm_energy_forces(positions_gpu, charges_gpu, cell, periodic, calc)

        @test isapprox(U_at, U_cpu; rtol = RTOL_F32)
        @test isapprox(Array(F_at), F_cpu; rtol = RTOL_F32)
    end
end
