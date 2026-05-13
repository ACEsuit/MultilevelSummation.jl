# Naive references

## `naive_energy` / `naive_energy_forces`

Direct `O(N²)` reference implementations used by the test suite and
useful for small-system sanity checks. Generic in the pair kernel and
boundary conditions.

```@docs
naive_energy
naive_energy_forces
```

## `MultilevelSummation.Reference` submodule

A naive 3D Ewald reference for fully periodic Coulomb lives in the
public `MultilevelSummation.Reference` submodule:

```julia
using MultilevelSummation.Reference: ewald_energy, ewald_energy_forces

U      = ewald_energy(positions, charges, cell; α=0.7, R_cut=10.0, k_cut=12.0)
U, F   = ewald_energy_forces(positions, charges, cell; α=0.7, R_cut=10.0, k_cut=12.0)
```

For a convenient auto-tuned call (it picks ``α``, ``R_\text{cut}``,
``k_\text{cut}`` from the cell extent and a target tolerance), see
`MultilevelSummation.Tune.ewald_reference` on the [Tune](tune.md) page.

It self-validates via the α-invariance property (Ewald's defining
property: total energy is independent of ``α`` when ``R_\text{cut}``
and ``k_\text{cut}`` are large enough). See `test/test_ewald.jl` for
the checks.

```@docs
MultilevelSummation.Reference.ewald_energy
MultilevelSummation.Reference.ewald_energy_forces
```
