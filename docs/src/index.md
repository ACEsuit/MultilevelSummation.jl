# MultilevelSummation.jl

A Julia package implementing the **Multilevel Summation Method (MSM)** of
Hardy, Wu, Phillips, Stone, Skeel & Schulten
(*J. Chem. Theory Comput.* **11**, 766–779, 2015) for fast evaluation of
long-range pair interactions.

This is currently an **early prototype**:

- Pure Julia, CPU only (kernel-shaped for a later `KernelAbstractions.jl`
  port).
- Dimensions ``d \in \{1,2,3\}``.
- Per-axis boundary conditions: `:open` / `:periodic` (mixed BC works).
- Pluggable kernels (`InversePower{N}`, `RationalDecay{N}`),
  splittings (`HardyC2Cubic` for Coulomb only so far) and interpolation
  bases (`CubicC1`).
- AtomsBase + AtomsCalculators interface.

The full implementation plan, including known limitations and the open
design questions, is in [`PLAN.md`](https://github.com/) at the repo root.

## Installation

```julia
using Pkg
Pkg.develop(path = "path/to/MultilevelSummation.jl")
```

## Quick start

A typical workflow at the **low-level (unit-free) API**:

```julia
using MultilevelSummation
using StaticArrays

# 5 random charges in a 4×4×4 box
positions = [SVector{3,Float64}(rand(3) .* 4) for _ in 1:5]
charges   = randn(5)
charges  .-= sum(charges) / 5         # neutralise

# Cell + periodicity
cell     = SMatrix{3,3,Float64}(4 * one(SMatrix{3,3,Float64}))
periodic = (true, true, true)

# MSM hyperparameters
calc = MSMCalculator(
    HardyC2Cubic(2.0, 4),             # cutoff a = 2, L = 4 levels
    CubicC1(),                        # cubic C¹ basis
    0.5,                              # finest grid spacing h = 0.5
)

U      = msm_energy(positions, charges, cell, periodic, calc)
U, F   = msm_energy_forces(positions, charges, cell, periodic, calc)
```

And the **AtomsBase / AtomsCalculators API**:

```julia
using AtomsBase, AtomsCalculators, Unitful

L = 4.0u"Å"
cell_vec = (SVector(L, 0u"Å", 0u"Å"),
            SVector(0u"Å", L, 0u"Å"),
            SVector(0u"Å", 0u"Å", L))
sys = periodic_system([
    Atom(:Na, SVector(1.0u"Å", 2.0u"Å", 0.5u"Å"); charge = +1.0),
    Atom(:Cl, SVector(2.5u"Å", 1.5u"Å", 3.0u"Å"); charge = -1.0),
], cell_vec)

U = AtomsCalculators.potential_energy(sys, calc)
F = AtomsCalculators.forces(sys, calc)
```

Units are stripped at the AtomsBase boundary; the numerical core is
unit-free.

## What MSM does, in one paragraph

The Coulomb (or other translation-invariant) pair kernel `K(r)` is split
into a short-range piece `K_0` with compact support `|r| ≤ a`, plus a
sequence of slowly varying pieces `K_1, …, K_L` interpolated on a nested
hierarchy of grids with spacings `h, 2h, 4h, …, 2^{L−1} h`. Particle
charges are scattered to the finest grid (anterpolation), restricted to
coarser grids, convolved with each level's `K_l` stencil, prolonged back
down, and gathered at the particles (interpolation). The result is the
long-range part of the potential. The short-range piece is summed
directly. The total energy is then assembled with a small self-correction
to remove the spurious self-interaction picked up by the long-range
pathway. See [Algorithm](algorithm.md) for a more detailed walk-through.

## What's in the package

| Module                              | What it contains                                   |
|-------------------------------------|----------------------------------------------------|
| Kernels                             | `InversePower{N}`, `Coulomb` alias, `RationalDecay{N}` |
| Splittings                          | `HardyC2Cubic` (Coulomb only at the moment)        |
| Basis                               | `CubicC1` (paper's piecewise cubic, C¹)            |
| Grid                                | `UniformGrid{D,T,Per,Sz}`, per-axis BC at the type level |
| Operators                           | `anterpolate!`, `interpolate!`, `restrict!`, `prolong!`, `grid_cutoff!`, `top_level!` |
| Naive O(N²) reference               | `naive_energy_forces` (any kernel)                 |
| `MultilevelSummation.Reference`     | `ewald_energy`, `ewald_energy_forces` — naive 3D Ewald |
| Assembly                            | `MSMCalculator`, `msm_energy`, `msm_energy_forces` |
| AtomsCalculators interface          | `potential_energy`, `forces`, `energy_forces`, `forces!` |
| `MultilevelSummation.Tune`          | `sweep`, `pareto_front`, `recommend`, `ewald_reference` |
