# Naive references

Direct `O(N²)` reference implementations used by the test suite and
useful for small-system sanity checks.

```@docs
naive_energy
naive_energy_forces
```

A naive 3D Ewald reference for fully periodic Coulomb is provided
under `test/refs/ewald.jl` (test-only, not part of the shipped package):

```julia
include(joinpath(pkgdir(MultilevelSummation), "test", "refs", "ewald.jl"))
using .EwaldRef: ewald_energy, ewald_energy_forces

U = ewald_energy(positions, charges, cell; α=0.7, R_cut=10.0, k_cut=12.0)
```

It self-validates via the α-invariance property (Ewald's defining
property: total energy is independent of ``α`` when ``R_\text{cut}`` and
``k_\text{cut}`` are large enough). See `test/test_ewald.jl` for the
checks.
