# Calculator

`MSMCalculator` bundles all MSM hyperparameters. It also serves as the
`AtomsCalculators.AbstractCalculator` for AtomsBase systems.

```@docs
MSMCalculator
msm_energy
msm_energy_forces
kernel_self_value
```

## AtomsCalculators interface

For a system `sys::AbstractSystem` (from `AtomsBase.jl`) and a calculator
`calc::MSMCalculator`:

```julia
AtomsCalculators.potential_energy(sys, calc)
AtomsCalculators.forces(sys, calc)
AtomsCalculators.forces!(F, sys, calc)
AtomsCalculators.energy_forces(sys, calc)
```

The calculator reads charges from the atom property whose name is
`calc.charge_property` (default `:charge`). Length units on positions
and cell vectors are stripped via `Unitful.ustrip`; the user is
responsible for ensuring positions, `h`, and `a` share a consistent
length unit. The numerical core is unit-free.
