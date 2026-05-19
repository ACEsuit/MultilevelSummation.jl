# End-to-end scaling study

Standalone driver that records wall-clock cost of
`msm_energy_forces` on realistic NaCl and H₂O fixtures across
system sizes from O(10²) to O(10⁵) atoms, on both the OhMyThreads
CPU path and the KernelAbstractions-CUDA path.

This is **publication preparation**, not regression tracking.
Regression baselines live in [`benchmark/`](../benchmark/);
hyperparameter sweeps live in [`tuning/`](../tuning/).

## What it measures

For each `(system, backend, N)` cell:

- one warm-up call (untimed),
- up to 5 timed calls of `msm_energy_forces`, captured as
  `time_min`, `time_median`, `time_mean`, `samples` in the CSV,
- `U` (total energy) and `U/N` (per-atom energy) for sanity tracking.

The MSM hyperparameters are the Pareto-optimal `(h, a)` values the
benchmark suite already uses, with the number of levels `L` grown
with the box so the top grid is always 1×1×1 — see *Why this gives
uniform accuracy* below.

| System | `h` (Å) | `a` (Å) | Box (Å)                   | Atoms                          |
|--------|--------:|--------:|---------------------------|--------------------------------|
| NaCl   |  2      |  4      | 4·`n_super` ∈ {8…128}     | 64, 512, 4 096, 32 768, 262 144 |
| H₂O    |  2      |  8      | {8, 16, 32, 64, 128}      | ~50, ~400, ~3 300, ~26 000, ~210 000 |

## Why this gives uniform accuracy

MSM error per pair is bounded by terms in `(h/a)^p` (interpolation,
basis order `p`) plus a splitting-error contribution that depends
on `a` and the kernel smoothness. Both are functions of `(a, h,
basis)` only — they don't depend on system size.

The number of levels `L` only sets the coarsest grid; it doesn't
introduce additional per-pair error. For a fully periodic box, the
natural choice is `L = log2(box/h) + 1`, which gives a 1×1×1 top
grid and lets the neutralising-background trick handle the
infinite periodic image. The script picks `L` this way for each
row.

For homogeneous systems, the energy and forces both scale linearly
with `N`, so the **relative** error stays constant across sizes.
The `U_per_atom` column in the CSV is the empirical witness: a
smoothly trending value across N confirms accuracy is uniform.

Constraint: `box/h` must be a power of 2 so `L` is integer and
each periodic axis bisects cleanly. The sweep above is chosen so
this holds.

## How to run

From the package root.

**One-time setup:**

```bash
julia --project=scaling -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

**CPU-only scan** (whichever thread count `-t` you choose is
recorded in the CSV):

```bash
julia -t auto --project=scaling scaling/scan.jl --label cpu-auto
julia -t 1    --project=scaling scaling/scan.jl --label cpu1
julia -t 128  --project=scaling scaling/scan.jl --label cpu128
```

**CPU + CUDA** (add CUDA to the scaling env once):

```bash
julia --project=scaling -e 'using Pkg; Pkg.add("CUDA")'
julia -t auto --project=scaling scaling/scan.jl --label a100
```

Replace `CUDA` with `AMDGPU` / `Metal` / `oneAPI` for the
corresponding hardware. The script auto-detects which framework is
installed and uses the first one it finds.

If you need to redirect the depot to a larger filesystem (CUDA
artifacts alone are ~2 GB), prepend `JULIA_DEPOT_PATH`:

```bash
JULIA_DEPOT_PATH="/path/to/big-disk-depot:$JULIA_DEPOT_PATH" \
  julia --project=scaling scaling/scan.jl --label a100
```

## Flags

| Flag                      | Default                          | Effect |
|---------------------------|----------------------------------|--------|
| `--label <name>`          | `<hostname>-t<threads>`          | Tag for the CSV filename. |
| `--systems nacl,h2o`      | `nacl,h2o`                       | Comma-separated subset of systems. |
| `--budget <seconds>`      | `60` (`MSM_SCALING_BUDGET_S`)    | Per-cell wall-clock budget. If a cell's median exceeds it, larger sizes for that backend are skipped. |
| `--skip-gpu`              | off                              | Force CPU-only even if a GPU framework is installed. |
| `--output <path>`         | `scaling/results/scaling-…csv`   | Override the output path. |
| `--no-accuracy-check`     | off                              | Skip the smallest-N CPU↔GPU `rtol = 1e-5` sanity comparison. |

## CSV columns

| Column        | Meaning |
|---------------|---------|
| `system`      | `nacl` or `h2o`. |
| `backend`     | `cpu` (OhMyThreads) or `gpu` (KA + array-type dispatch). |
| `label`       | The `--label` argument; identifies the run. |
| `threads`     | `Threads.nthreads()` at run start (1 for GPU rows too — Julia thread count). |
| `framework`   | `OhMyThreads`, `CUDA`, `AMDGPU`, `Metal`, or `oneAPI`. |
| `N`           | Atom count for this cell. |
| `box`         | Cubic-box side length (Å). |
| `h, a, L`     | MSM hyperparameters used. |
| `time_min`, `time_median`, `time_mean` | Seconds, across `samples` timed runs. |
| `samples`     | Number of timed runs (0 if budget-skipped). |
| `U`           | Total energy (kJ/mol-equivalent, see `MultilevelSummation` unit convention). |
| `U_per_atom`  | `U / N`. Empirical witness of uniform-accuracy claim. |
| `note`        | Free-text annotation (e.g. `"budget gate triggered after 16"`). |

## Memory notes (A100 40 GB)

The largest cell is H₂O at box = 128 Å (~210 000 atoms) or NaCl at
n_super = 32 (262 144 atoms). With `h = 2`, the fine grid is 64³ ≈
2 MB per level; the L = 7 hierarchy totals well under 100 MB.
Particle and grid buffers stay comfortably within 40 GB. The
budget gate trips long before memory does.

## Deferred / explicitly out of scope

- **Plotting.** A `plot.jl` companion that aggregates multiple CSVs
  into log-log plots can be added on top later; the data here is
  the deliverable.
- **Ewald comparison column.** Planned for the eventual paper.
  Ewald CPU runtime is prohibitive at `N ≥ 10⁴` (`O(N²)` direct
  sums), so this requires a fast-Ewald comparison library.
- **Per-stage breakdown** (anter / interp / restrict / prolong /
  convolve / short-range). Useful for the paper; not exported here
  yet — only end-to-end wall clock.
- **Energy-only timings.** Forces dominate so we measure
  `msm_energy_forces` only.

## H₂O accuracy caveat (until upstream lands)

Until [JuliaMolSim/NeighbourLists.jl#38](https://github.com/JuliaMolSim/NeighbourLists.jl/pull/38)
merges, the KA short-range loop misses ~10% of pairs on TIP3P H₂O
configurations (the cell list mishandles atoms placed outside
`[0, L)` along a periodic axis). The script still runs to completion,
but the smallest-N CPU↔GPU spot check on H₂O may report a
relative error above `rtol = 1e-5`. The energies are still
self-consistent within the KA path — both CPU-KA and GPU-KA agree to
roundoff — but they disagree with the legacy `msm_energy` (which
uses the naive O(N²) loop) at the affected sizes. This is tracked
as a known issue in `PRIORITIES.md` T1 follow-ups.
