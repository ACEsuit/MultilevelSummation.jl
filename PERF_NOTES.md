# Performance notes

Record of performance work on the `perf` branch and what's queued.

> **Baseline note (post-`tune` consolidation).**
> The benchmark suite was migrated from random-system fixtures
> (`_make_periodic_system`, N=32) to realistic NaCl/H2O configurations
> from `Tune.build_nacl` / `Tune.build_h2o` at the Pareto-optimal
> `(h, a, L)` from the tuning sweeps. The `PkgBenchmark.judge`
> comparisons against pre-consolidation commits below are therefore
> **no longer apples-to-apples** for the `end_to_end`, `scaling_N`,
> and `precision` groups. The `operators` and `wrap_mode` groups still
> drive grids directly with `StableRNG` charges and can be compared
> across the consolidation.

## Status

Five rounds of optimisation have landed on `perf`, taking `msm_energy`
in 3D from ~1× (baseline `main`) to **~3× faster** with comparable
allocation reductions, and the per-particle anter/interpolation ops
from heavy boxing to bounds-check-free, allocation-free tight loops
(~100× microbench speedup).

Cumulative `PkgBenchmark.judge(perf, main)` highlights:

| Operation | perf / main | Speedup |
|---|---|---|
| `msm_energy_D=3` | 0.32 | 3.1× |
| `msm_energy_forces_D=3` | 0.28 | 3.6× |
| `grid_cutoff!_level1` (n=8, pow2) | 0.30 | 3.3× |
| `grid_cutoff!_pow2_n=16` | 0.25 | 4.0× |
| `grid_cutoff!_nonpow2_n=12` | 0.33 | 3.0× |
| `msm_energy_pow2_n=16` | 0.34 | 2.9× |
| `msm_energy_nonpow2_n=12` | 0.41 | 2.4× |
| `anterpolate!` / `interpolate!` | 0.01 / 0.01 | ~100× |
| `interpolate_grad!` | below measurement floor | — |
| `top_level!` | 0.86–0.92 | 1.1× |
| `prolong!` | 0.76–1.13 | run-to-run noise around 1× |
| `restrict!` | **1.21–1.25** :x: | **~21 % slower** (unexplained) |

All 3642 tests pass on `perf`.

## What's done

1. **PBC into the type parameter.** `UniformGrid{D,T}` → `UniformGrid{D,T,Per}`,
   where `Per::NTuple{D,Bool}` is the per-axis periodicity carried at the
   type level. Removed the runtime branch on `g.periodic[α]` inside
   `wrap_index` and the other operators. A `getproperty` shim keeps
   `g.periodic` working as before so no call site needed to change.

2. **Grid size into the type parameter.** `UniformGrid{D,T,Per}` →
   `UniformGrid{D,T,Per,Sz}`, where `Sz::NTuple{D,Int}` is the per-axis
   extent at the type level. The `size` field is gone; `getproperty`
   returns `Sz`. The compiler now sees `mod(idx, n)` and `idx < n` with
   `n` an integer literal — `mod` lowers to a multiply-high sequence.
   `coarser_grid` propagates `Per` and produces a new `Sz`; the
   level-hierarchy storage in `_msm_compute` became `Vector{Any}` (each
   level is a different concrete `Sz` type), with one dynamic dispatch
   per call boundary that turned out to be ~free.

3. **Bounds-check-free `_convolve!`.** `gridcutoff.jl`'s convolution is
   now a `@generated` function specialised on `(Per, Sz)`. The per-axis
   inner loop is unrolled; for periodic axes it iterates the full
   `-smax:smax` range with a `mod`-wrap, and for open axes it clips the
   *loop bounds* to `max(-smax, -(m-1)):min(smax, Sz-m)` so the source
   index is in-range by construction (no per-iteration check).

4. **Type-stable `anterpolate!` / `interpolate!` / `interpolate_grad!`.**
   The three particle ↔ grid operators now use a barrier-function pattern
   (`Val(support_radius(basis))` lifted to type parameter `S` for the
   `_impl!` body) plus two small `@generated` helpers
   (`_tensor_basis_value`, `_tensor_basis_gradient`) for the per-axis
   tensor product. Eliminated `bv::ANY` and the abstract inner tuples in
   `ϕ_per_axis`, unlocking SIMD and removing the heap allocations the
   boxed accumulator was causing.

5. **Power-of-2 bitmask wrap for periodic axes.** In the same `@generated`
   blocks (`wrap_index`, `_convolve!`), when `Sz[α]` is a power of two,
   the generator emits

       ((idx + n) & (n - 1)) + 1

   instead of `mod(idx, n) + 1`. The benchmark suite was extended with
   a `wrap_mode` group that pairs a pow2 size (n=16) and a non-pow2 size
   (n=12) so the bitmask win is visible in `PkgBenchmark.judge` output.
   The pow2 path is **~20 % faster than the non-pow2 path** on both
   `grid_cutoff!` (0.25 vs 0.33) and full `msm_energy` (0.34 vs 0.41) —
   on top of all the prior gains.

## Outstanding: `restrict!` regression

`restrict!` is consistently 21–25 % slower on `perf` than on `main`,
both on standalone and end-to-end benchmarks. Investigation in
`profile/3_restrict.jl` ruled out two leading hypotheses:

- **Internal type instability**: `code_warntype` shows `restrict!`'s body
  is fully type-stable on `perf` — `bv::Float64`, `s::Core.Const(2)` via
  const-prop on `support_radius`, no `::Any` anywhere.
- **`Vector{Any}` dispatch overhead in `_msm_compute`**: a direct call
  to `restrict!` and a call through `grids_any::Vector{Any}` measure
  within ~50 ns of each other (98.0 μs each) — totally lost in noise.

So the regression is a subtler codegen-level interaction (likely between
the unchanged `restrict!` body and the new `@generated wrap_index`, or
the per-level concrete `Sz` types). Not yet diagnosed. Suggested next
investigation: `Cthulhu.@descend` or side-by-side `@code_native` diff
between branches. Tools and pointers are in `profile/3_restrict.jl §C`.

For typical 3D MD workloads `restrict!` is a small fraction of total
time, so the net effect is comfortably positive — `restrict!` is just
the only operator that didn't benefit from this round.

## Suggested next perf items (not actively pursued)

In rough order of expected payoff:

1. **Apply the bounds-free `@generated` treatment to `restrict!` and
   `prolong!`.** Same pattern as `_convolve!` — clip the loop range per
   destination grid point for open axes, full range with bitmask/mod
   for periodic. Probably resolves the `restrict!` regression and lets
   both transfer ops benefit from the type-parameter machinery.

2. **Diagnose the residual `restrict!` regression with Cthulhu/asm
   diff.** Even before (1) lands, knowing the root cause is useful.

3. **GPU port via `KernelAbstractions.jl`.** The current CPU code is
   "kernel-shaped" and the per-axis dispatch is at codegen rather than
   runtime, so the port should be mostly mechanical. `grid_cutoff!` is
   the obvious headline kernel — it's already a perfect output-parallel
   stencil, expected 100–500× on a modern GPU.

4. **Stencil tiling / shared-memory layout for `grid_cutoff!`** (CPU
   *and* GPU). The current loop nest is column-major-correct but
   doesn't tile for cache. A small tile size that fits L1 + halo
   should give another factor on big grids.

5. **Multithreading via `@threads` for the outer `m` loop of
   `_convolve!`** and the particle loops in anter/interpolate. Each is
   embarrassingly parallel.

6. **`Float32` SIMD.** Float64 currently wins/ties Float32 in our
   benchmarks because the bottleneck was allocation/memory, not flops.
   With everything now type-stable and allocation-free, Float32 should
   pull ahead if the inner loops are wide enough — worth re-measuring.
