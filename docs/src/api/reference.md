# `MultilevelSummation.Reference` submodule

The `Reference` submodule hosts the two absolute baseline
implementations used by the test suite and the tuning scripts:

- `naive_energy` / `naive_energy_forces` — direct `O(N²)` sum,
  generic in the pair kernel and boundary conditions.
- `ewald_energy` / `ewald_energy_forces` — naive 3D Ewald reference
  for fully periodic Coulomb.

```julia
using MultilevelSummation.Reference: naive_energy, naive_energy_forces
using MultilevelSummation.Reference: ewald_energy, ewald_energy_forces
```

## `naive_energy` / `naive_energy_forces`

Useful for small-system sanity checks. Periodic axes are handled by
truncating the lattice-image sum to a user-supplied `R_cut`.

```@docs
MultilevelSummation.Reference.naive_energy
MultilevelSummation.Reference.naive_energy_forces
```

## `ewald_energy` / `ewald_energy_forces`

```julia
U     = ewald_energy(positions, charges, cell; α=0.7, R_cut=10.0, k_cut=12.0)
U, F  = ewald_energy_forces(positions, charges, cell; α=0.7, R_cut=10.0, k_cut=12.0)
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
