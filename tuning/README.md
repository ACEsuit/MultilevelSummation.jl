# Hyperparameter tuning

Scripts that sweep MSM hyperparameters `(h, a, L)` on realistic test
systems and compare against an absolute reference (naive 3D Ewald).
Output is a table on stdout plus a CSV next to the script.

The scripts are **thin callers** of the public
[`MultilevelSummation.Tune`](../docs/src/api/tune.md) submodule. If you
want programmatic sweeps from your own code rather than running these
scripts, call `Tune.sweep` directly — see `docs/src/api/tune.md` or the
examples page.

## What's here

| Script | System | Status |
|---|---|---|
| [`tune_NaCl.jl`](tune_NaCl.jl) | rock-salt NaCl supercell, ±1 ion charges | shipped |
| [`tune_NaCl_perturbed.jl`](tune_NaCl_perturbed.jl) | rock-salt NaCl with σ ≈ 0.1 Å thermal displacements | shipped |
| [`tune_H2O.jl`](tune_H2O.jl) | TIP3P-like liquid-water box (Poisson-disk oxygens, random orientations) | shipped |

## How to run

From the package root:

```bash
julia --project=tuning tuning/tune_NaCl.jl
```

On first invocation the tuning environment needs to be instantiated
(it just dev-links `MultilevelSummation`; `SpecialFunctions` and
`StaticArrays` flow in as indirect deps):

```bash
julia --project=tuning -e '
    using Pkg
    Pkg.develop(path = ".")
    Pkg.instantiate()'
```

## Configuration

Edit the three constants at the top of `tune_NaCl.jl`:

- `N_SUPER_LIST` — supercell sizes `n` such that `N = 8·n³`. Default is
  `(3, 4, 5, 6, 8, 10, 12)` ⇒ `N ∈ {216, 512, 1000, 1728, 4096, 8000,
  13824}`. **Comment the larger ones out for a quick first run** —
  Ewald is naive O(N²) and runtime scales accordingly.
- `H_VALUES` — finest grid spacings `h` in Å. Must divide the supercell
  side `n·a_lat = 4n` cleanly; invalid combinations are skipped by
  `Tune.sweep`.
- `A_VALUES` — short-range cutoffs `a` in Å.

`Tune.sweep` auto-picks `L` from the largest power of two that divides
`n_grid = box/h`, capped at 6 levels. With `L_strategy = :all` the
script iterates `L = 2 … L_max` for each `(h, a)`; with `:max` it
returns only the largest valid `L` per `h`.

## Programmatic use

```julia
using MultilevelSummation
using MultilevelSummation.Tune

U_ref = Tune.ewald_reference(positions, charges, cell)     # tol=1e-9 by default
results = Tune.sweep(positions, charges, cell, periodic;
                    reference  = U_ref,
                    h_values   = (0.5, 1.0, 2.0),
                    a_values   = (2.0, 4.0, 8.0),
                    L_strategy = :all)

# Inspect the trade-off
front = Tune.pareto_front(results)
best  = Tune.recommend(results; max_rel_err = 1e-3)
```

`results` is a `Vector{SweepResult}` with fields
`(h, a, L, n_grid, energy, rel_err, t_msm)`. The same data the script
writes to CSV.

## Output

- **stdout**: one block per supercell size — Ewald reference + a row
  per `(h, a, L)` combination with relative energy error and MSM
  wall-clock time.
- **CSV**: `tuning/tune_NaCl_results.csv`, columns
  `N,n_super,box,h,n_grid,a,L,rel_err,t_msm,t_ewald,U_ref`. Suitable
  for plotting accuracy-vs-cost or quick sanity checks in another tool.

The CSV file is gitignored.

## Cost notes

The MSM side is cheap (≤ seconds even for the largest sweep entry).
The bottleneck is the Ewald reference because it's naive — each
supercell pays one Ewald call. Rough wall-clock on an M-class CPU:

| `n_super` | `N` | Ewald (s) |
|---|---|---|
| 3 | 216 | < 1 |
| 4 | 512 | ~1 |
| 6 | 1728 | ~10 |
| 8 | 4096 | ~50 |
| 10 | 8000 | ~3 min |
| 12 | 13824 | ~10 min |

If you only need the small/medium range, drop the last few entries
from `N_SUPER_LIST`. A neighbour-list Ewald would speed this up
substantially but isn't needed for one-time tuning.
