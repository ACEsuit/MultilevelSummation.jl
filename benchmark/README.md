# MLSum benchmarks

The benchmark suite uses [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
and is invoked via [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl).

## Quick run

```bash
julia --project=benchmark -e '
    using PkgBenchmark
    results = benchmarkpkg("MLSum")
    export_markdown("benchmark/results.md", results)
'
```

Takes ≈ 1 minute on a laptop. Writes a markdown report to
`benchmark/results.md`.

## Compare two revisions

```bash
julia --project=benchmark -e '
    using PkgBenchmark
    judgement = judge("MLSum", "HEAD", "main")
    export_markdown("benchmark/judge.md", judgement)
'
```

Runs the suite on both revisions and reports the speedup/regression
ratios. Requires a clean git tree.

## Suite layout

The suite (`benchmark/benchmarks.jl`) is organised into four groups:

- **`end_to_end`** — `msm_energy` and `msm_energy_forces` across
  dimensions, plus a naive O(N²) baseline at the same N.
- **`operators`** — per-operator microbenchmarks (`anterpolate!`,
  `restrict!`, `prolong!`, `grid_cutoff!`, `top_level!`, `interpolate!`,
  `interpolate_grad!`). Useful for profiling individual phases.
- **`scaling_N`** — `msm_energy` vs the periodic naive sum at several
  particle counts. Highlights the crossover behaviour.
- **`precision`** — `msm_energy` at `Float64` vs `Float32` to verify
  that the type-generic core actually exercises `Float32` and to compare
  cost.

Default parameters are chosen modest (`N = 32`, `n_fine = 8`, `L = 4`)
so the suite is CI-friendly. For local profiling bump `N_DEFAULT`,
`L_DEFAULT`, or add larger sizes to `scaling_N`.

## Notes from the latest run

These numbers (Apple M-class CPU, Julia 1.12, prototype quality, no
optimisation) are roughly what to expect:

| Configuration                 | Time   | Allocations |
|-------------------------------|--------|------------|
| `msm_energy` D=3, N=32        | ~6.5 ms | ~53k        |
| `naive_energy` periodic, N=32 | ~30 μs  | 0          |
| `naive_energy` periodic, N=64 | ~130 μs | 0          |
| `anterpolate!` (N=32, d=3)    | ~240 μs | ~28k        |
| `grid_cutoff!` level 1        | ~5 ms   | ~3          |
| `top_level!` (1×1×1 grid)     | ~17 μs  | 0          |

Headline observations:

- **Naive O(N²) is much faster than MSM for small N.** Expected — MSM's
  multilevel machinery is overhead at this size. The crossover for
  Coulomb in 3D is typically several thousand atoms.
- **`grid_cutoff!` dominates the MSM time** (~5 ms out of ~6.5 ms total
  for D=3). It's the obvious target for the eventual
  `KernelAbstractions.jl` port.
- **Anter/interpolation allocate heavily** (~28k–106k allocations per
  call). Each particle currently builds per-axis basis-value tuples
  with `ntuple(..., Val(D))` which materialise on the heap inside the
  outer loop. Worth flattening to scalar locals in the hot path.
- **`Float32` is not faster than `Float64`** at this size — the
  bottleneck is allocation/memory traffic, not floating-point throughput.

Reducing allocations in the per-particle operators is the most
productive perf target before the GPU port.
