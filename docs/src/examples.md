# Examples

Concrete, runnable examples that exercise the major code paths.

## Open-BC Coulomb, 5 charges in 3D

```@example open3d
using MultilevelSummation
using StaticArrays
using Random
Random.seed!(42)

positions = [SVector{3,Float64}((rand(3) .* 2)...) for _ in 1:5]
charges   = randn(5) ./ sqrt(5)

cell     = zero(SMatrix{3,3,Float64})            # unused for open BC
periodic = (false, false, false)

# Baseline: direct O(N²) Coulomb
K = Coulomb()
U_naive = naive_energy(positions, charges, cell, periodic, K)

# MSM with a fairly small a — splitting error is the dominant approximation here.
calc = MSMCalculator(HardyC2Cubic(2.0, 3), CubicC1(), 0.2)
U_msm = msm_energy(positions, charges, cell, periodic, calc)

(U_naive, U_msm, abs(U_msm - U_naive) / abs(U_naive))
```

## Convergence in `a`

Holding `h` fixed and growing `a` should reduce the splitting error
monotonically.

```@example convergence
using MultilevelSummation, StaticArrays, Random
Random.seed!(0xC0FFEE)

positions = [SVector{3,Float64}((rand(3) .* 2)...) for _ in 1:8]
charges   = randn(8); charges .-= sum(charges)/length(charges)
cell, periodic = zero(SMatrix{3,3,Float64}), (false, false, false)

U_naive = naive_energy(positions, charges, cell, periodic, Coulomb())

for a in (0.8, 1.6, 3.2, 6.4)
    calc = MSMCalculator(HardyC2Cubic(a, 3), CubicC1(), 0.2)
    U = msm_energy(positions, charges, cell, periodic, calc)
    @show a, abs(U - U_naive) / abs(U_naive)
end
```

## Periodic 3D Coulomb

```@example periodic
using MultilevelSummation, StaticArrays, Random
Random.seed!(7)

h, L = 0.5, 4                        # 8 fine points per axis ⇒ top is 1×1×1
cell_L = h * 8
cell = SMatrix{3,3,Float64}(cell_L * one(SMatrix{3,3,Float64}))
periodic = (true, true, true)

N = 8
positions = [SVector{3,Float64}((rand(3) .* cell_L)...) for _ in 1:N]
charges   = randn(N); charges .-= sum(charges)/N

calc = MSMCalculator(HardyC2Cubic(2.0, L), CubicC1(), h)
U, F = msm_energy_forces(positions, charges, cell, periodic, calc)
(U, sum(F))                          # ΣF ≈ 0 expected
```

## Through the AtomsCalculators interface

```@example atoms
using MultilevelSummation, AtomsBase, AtomsCalculators, Unitful, StaticArrays

L = 4.0u"Å"
cell_vec = (SVector(L, 0u"Å", 0u"Å"),
            SVector(0u"Å", L, 0u"Å"),
            SVector(0u"Å", 0u"Å", L))
sys = periodic_system([
    Atom(:Na, SVector(1.0u"Å", 2.0u"Å", 0.5u"Å"); charge = +1.0),
    Atom(:Cl, SVector(2.5u"Å", 1.5u"Å", 3.0u"Å"); charge = -1.0),
], cell_vec)

calc = MSMCalculator(HardyC2Cubic(2.0, 4), CubicC1(), 0.5)
ef = AtomsCalculators.energy_forces(sys, calc)
(ef.energy, ef.forces[1])
```

## Mixed BC: periodic in xy, open in z

```@example mixed
using MultilevelSummation, StaticArrays, Random
Random.seed!(11)

# 4×4 periodic in xy, open in z
L_xy   = 4.0
cell   = SMatrix{3,3,Float64}([L_xy 0 0; 0 L_xy 0; 0 0 1.0])
periodic = (true, true, false)

positions = [SVector(2.0, 1.5, 0.3), SVector(0.5, 3.2, -0.4)]
charges   = [1.0, -1.0]

calc = MSMCalculator(HardyC2Cubic(2.0, 3), CubicC1(), 0.5)
msm_energy(positions, charges, cell, periodic, calc)
```

## Naive Ewald reference

A naive 3D Ewald reference lives in the public
`MultilevelSummation.Reference` submodule. Used in tests for
periodic-Coulomb accuracy checks and as the reference inside
`MultilevelSummation.Tune.sweep`:

```julia
using MultilevelSummation.Reference: ewald_energy, ewald_energy_forces

U      = ewald_energy(positions, charges, cell; α=0.7, R_cut=10.0, k_cut=12.0)
U, F   = ewald_energy_forces(positions, charges, cell; α=0.7, R_cut=10.0, k_cut=12.0)
```

For auto-tuned parameters use the `Tune` wrapper:

```julia
using MultilevelSummation.Tune
U = Tune.ewald_reference(positions, charges, cell; tol = 1e-9)
```

Both perform the standard erfc / reciprocal-Gaussian split and self-
validate via α-invariance in `test/test_ewald.jl`.

## Programmatic hyperparameter sweep

```@example tune
using MultilevelSummation
using MultilevelSummation.Tune
using StaticArrays, Random
Random.seed!(0xABCD)

L = 8.0; cell = SMatrix{3,3,Float64}(L * one(SMatrix{3,3,Float64}))
periodic = (true, true, true)
positions = [SVector{3,Float64}((rand(3) .* L)...) for _ in 1:32]
charges = randn(32); charges .-= sum(charges) / length(charges)

U_ref = Tune.ewald_reference(positions, charges, cell)
results = Tune.sweep(positions, charges, cell, periodic;
                    reference  = U_ref,
                    h_values   = (0.5, 1.0),
                    a_values   = (1.0, 2.0),
                    L_strategy = :all)
length(results), Tune.recommend(results; max_rel_err = 0.1)
```

The shipped `tuning/tune_NaCl.jl` is a thin caller of this same API on
realistic NaCl supercells — see `tuning/README.md` for how to run it.
