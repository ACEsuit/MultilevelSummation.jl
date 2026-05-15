# Internals notes

A small collection of useful internal notes that aren't part of the
public API but are good to know when contributing.

## Layer separation

The package is structured in two layers:

1. **Numerical core** (`src/core.jl` and most of `src/*.jl`). Pure
   numerical routines on plain arrays of `SVector{D,T}` / `T`. No units,
   no AtomsBase, no calculator object outside `MSMCalculator` itself.
   This is what gets exercised by `msm_energy(...)` /
   `msm_energy_forces(...)` and what will be ported to
   `KernelAbstractions.jl`.
2. **AtomsCalculators wrapper** (`src/calculator.jl`). A thin layer that
   extracts data from an `AtomsBase.AbstractSystem`, strips units, and
   delegates to the core. Most of the per-call overhead in this layer is
   the conversion, so direct numerical-core users avoid it.

## Kernel-shaped code

Each operator (`anterpolate!`, `restrict!`, `prolong!`, `grid_cutoff!`,
`top_level!`, `interpolate!`, `interpolate_grad!`) is written as a single
loop nest with no method dispatch inside the inner loops. Allocations
live at the call site. This is preparation for the eventual
`KernelAbstractions.jl` port.

## Build hierarchy notes

`build_grid_hierarchy` derives the level-1 grid from the cell + positions:

- **Periodic axes**: `n_α = round(L_α / h)`. Required: `n_α` divisible by
  ``2^{L-1}``.
- **Open axes**: `n_α = ceil((bbox_α + 2·pad) / h)`, rounded up to a
  multiple of ``2^{L-1}``. `pad = support_radius(basis) * h`.

Each subsequent coarser level halves the extent along every axis. For
fully periodic systems with the top grid reduced to `1×1×1`,
`apply_neutralising_background!` zeros that single charge if the
splitting requires it (Coulomb does).

## Stencil width and the periodic-grid constraint

The grid-cutoff stencil at level `l` has half-width

```math
s_\text{max} = \lceil 2a / h_1 \rceil
```

in grid points (independent of `l`!), because the kernel ``K_l`` has
compact support ``2^l a`` and the grid spacing at level `l` is
``2^{l-1} h_1``. For a periodic axis with extent `n`, we need
``2 s_\text{max} + 1 \le n`` to avoid the stencil wrapping around and
causing self-image aliasing. The constructor of `MSMCalculator`
does not yet check this — be aware when choosing small grids.

## Self-correction

The long-range pathway accumulates a spurious self-interaction
``q_i^2 \cdot K_\text{long}(0)`` for each particle (via the
anterpolation–interpolation chain). The energy formula in `msm_energy`
explicitly subtracts ``\tfrac{1}{2} K_\text{long}(0) \sum_i q_i^2``
where ``K_\text{long}(0) = \sum_{l=1}^L K_l(0)`` is computed by
`kernel_self_value(splitting, Val(D), T)`. The self term has no spatial
gradient, so it contributes nothing to forces.

## Current limitations

- Cells are required to be **orthorhombic** in the naive reference, the
  grid hierarchy, and the Ewald reference. General triclinic cells need
  more careful image-shift bookkeeping; the data structures already
  carry a full ``D \times D`` cell matrix to allow this extension.
- Only `HardyC2Cubic` is implemented, valid for the Coulomb kernel.
  Splittings for `InversePower{N\ne 1}` and `RationalDecay{N}` need a
  new ``\gamma`` matched to ``s^{-N/2}`` and are TBD.
- For open BC the grid origin tracks the particle bounding box, so the
  energy is only piecewise-smooth in individual particle positions
  (continuous and exactly translation-invariant under simultaneous
  shifts of all particles, but FD-vs-analytic comparisons match only
  when the grid is fixed). Periodic BC has a cell-fixed grid and
  matches FD to machine precision.

See [`PLAN.md`](https://github.com/ACEsuit/MultilevelSummation.jl/blob/main/PLAN.md)
for the architectural design contract and
[`PRIORITIES.md`](https://github.com/ACEsuit/MultilevelSummation.jl/blob/main/PRIORITIES.md)
for the live task list / open design questions.
