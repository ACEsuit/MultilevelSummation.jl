# Hyperparameter tuning

Scripts that sweep MSM hyperparameters `(h, a, L)` on realistic test
systems and compare against an absolute reference (naive 3D Ewald).
Output is a table on stdout, a CSV, and PNG plots next to the script.

The scripts are **thin drivers** over the public
[`MultilevelSummation.Tune`](../docs/src/api/tune.md) submodule, which
hosts both the sweep orchestration and the system builders
(`Tune.build_nacl`, `Tune.build_h2o`). The same builders are used by
the benchmark suite (`benchmark/benchmarks.jl`) and the test suite
(`test/test_systems.jl`, `test/test_tune.jl`) as common realistic
fixtures.

## What's here

| Script | System | Notes |
|---|---|---|
| [`tune_NaCl.jl`](tune_NaCl.jl) | rock-salt NaCl supercell, ±1 ion charges | `SIGMA` constant at the top controls the Gaussian thermal displacement (Å). `SIGMA = 0` recovers a perfect lattice. |
| [`tune_H2O.jl`](tune_H2O.jl) | TIP3P-like liquid-water box | Oxygens placed by Poisson-disk rejection (Bridson + uniform fallback); molecules given uniform random SO(3) orientations. |
| [`plotting.jl`](plotting.jl) | shared plotting helper | `plot_sweep(rows, dir, prefix; subtitle)` consumed by both tune scripts. Lives in the tuning environment only — `Plots.jl` is not a core dep. |

## How to run

From the package root:

```bash
julia -t auto --project=tuning tuning/tune_NaCl.jl
julia -t auto --project=tuning tuning/tune_H2O.jl
```

The hot operators (Ewald reference and MSM convolutions) are
multi-threaded via OhMyThreads — `-t auto` (or any explicit thread
count) substantially reduces the `(h=0.5, a=8)` bottleneck at the
largest sweep entries.

On first invocation the tuning environment needs to be instantiated
(it dev-links `MultilevelSummation` and pulls in `Plots`):

```bash
julia --project=tuning -e '
    using Pkg
    Pkg.develop(path = ".")
    Pkg.instantiate()'
```

## Configuration

Constants at the top of each script:

- `tune_NaCl.jl`:
  - `SIGMA` — Gaussian thermal displacement per Cartesian (Å).
    Default `0.1`; set to `0.0` for the perfect lattice.
  - `N_SUPER_LIST` — supercell sizes; `N = 8·n³` ions per entry.
- `tune_H2O.jl`:
  - `BOX_LENGTHS` — cubic box sides in Å.
  - `DENSITY` — molecules / Å³ (default `0.0334`, ≈ 1 g/cm³).
  - `D_MIN_OO` — minimum O–O distance for Poisson-disk sampling.
- both:
  - `H_VALUES`, `A_VALUES` — `(h, a)` sweep grid.
  - `RNG_SEED` — pinned seed for reproducibility.

`Tune.sweep` (called by `run_system_sweep`) auto-picks `L` from the
largest power of two that divides `n_grid = box/h`, capped at 6 levels.
With `L_strategy = :all` the script iterates `L = 2 … L_max` for each
`(h, a)`.

## Programmatic use

```julia
using MultilevelSummation
using MultilevelSummation.Tune

# Build a system, run the sweep, get one NamedTuple row per (size, h, a, L).
builder = n -> Tune.build_nacl(n; σ = 0.1)
rows = Tune.run_system_sweep(builder, (2, 3, 4); size_label = "n_super")

# Text summary (Pareto, recommendations, scaling slopes).
Tune.print_summary(rows)

# Or call the lower-level building blocks directly:
positions, charges, cell, periodic = Tune.build_nacl(3; σ = 0.1)
U_ref = Tune.ewald_reference(positions, charges, cell)
results = Tune.sweep(positions, charges, cell, periodic;
                     reference = U_ref, L_strategy = :all)
front = Tune.pareto_front(results)
best  = Tune.recommend(results; max_rel_err = 1e-3)
```

## Output

- **stdout**: per-system block — Ewald reference + a row per
  `(h, a, L)` combination — followed by the summary (Pareto front,
  threshold-recommended settings, error/cost scaling slopes).
- **CSV**: `tuning/tune_{NaCl,H2O}_results.csv`, columns
  `size,N,box,h,n_grid,a,L,rel_err,t_msm,t_ewald,U_ref`. Gitignored.
- **PNGs**: `tuning/plots/{nacl,h2o}_{accuracy_vs_cost,error_vs_h,cost_vs_N}.png`.
  Gitignored.

## Cost notes

For the NaCl sweep with σ = 0.1 Å, n_super ∈ {3, 4, 5, 6, 8}, the
bottleneck is the `(h=0.5, a=8)` corner of the sweep at the largest
`n_super` — that single setting dominates total wall-clock. Ewald
itself is sub-second even at n_super = 8 (4096 ions). Rough total
wall-clock on an M-class CPU:

| `N_SUPER_LIST`               | wall-clock |
|------------------------------|-----------|
| `(3, 4, 5, 6)`               | ~30 s     |
| `(3, 4, 5, 6, 8)`            | ~15 min   |
| `(3, 4, 5, 6, 8, 10, 12)`    | ≳ 30 min  |

For the H2O sweep with `BOX_LENGTHS = (16, 20, 24)`, total wall-clock
is ~3 min, dominated by the box=24 `(h=0.5, a=8)` corner.

If you only need an overview, narrow the lists or drop the largest
entries. A neighbour-list Ewald would speed up the reference
substantially but isn't needed for one-time tuning.
