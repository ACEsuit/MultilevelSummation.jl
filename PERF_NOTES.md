# Performance follow-ups

Live notes on performance work, including what's done and what's queued.

## Done so far (on the `perf` branch)

1. **PBC into the type parameter** (`UniformGrid{D,T,Per,Sz}`). Removed
   the runtime branch on `g.periodic[α]` in `wrap_index` and adjacent
   call sites.

2. **Grid size into the type parameter.** Per-axis extent `n_α` is now
   a compile-time constant via the `Sz` type parameter, so any code
   that does `mod(idx, n_α)` or `idx < n_α` sees an integer literal
   instead of a runtime struct-field load. For *general* integer
   constants, the compiler lowers `mod` to a multiply-high sequence
   (≈ 5–10 cycles) — already a clean win over a runtime division.

3. **Bounds-check-free `_convolve!`** (`gridcutoff.jl`). `@generated`
   per `(Per, Sz)`. The per-axis stencil loop is unrolled, the source
   index is bounds-clipped at the *loop bounds* (open axes) or wrapped
   by `mod` (periodic axes) — no per-iteration check inside the inner
   loop.

4. **Type-stable `anterpolate!` / `interpolate!` / `interpolate_grad!`**
   via a barrier function (`Val(support_radius(basis))` lifted into a
   type parameter `S` for the `_impl!` body) plus two small `@generated`
   helpers (`_tensor_basis_value`, `_tensor_basis_gradient`) for the
   per-axis tensor product. Eliminates `bv::ANY` and the abstract
   inner tuples in `ϕ_per_axis`, unlocking SIMD and removing boxing
   allocations.

Cumulative result vs `main` (judged via `PkgBenchmark.judge`):

- `msm_energy_D=3`: **~3× faster**
- `msm_energy_forces_D=3`: **~3.4× faster**
- `anterpolate!` / `interpolate!`: **~100×** (was bottlenecked by
  type instability + boxing)
- `grid_cutoff!`: ~3.2× faster
- Outstanding regressions: `restrict!` ≈ 1.24×, `prolong!` ≈ 1.13×
  (from `Vector{Any}` storage of the level hierarchy — one dynamic
  dispatch per call into transfer ops). For 3D-MD-typical workloads
  these are a small fraction of total time so the trade-off is net
  positive.

## Next: power-of-2 grid sizes (cheaper periodic wrap)

### Why it should help

With `Sz` already in the type, the compiler sees `mod(idx, n_α)` where
`n_α` is an integer literal. For a *general* constant `n`, this lowers
to a multiply-high sequence — already faster than runtime division but
still several cycles. For a **power-of-two** constant `n = 2^k`, the
operation collapses to a single bitwise AND:

```
mod(idx, n)   →   idx & (n - 1)
```

That's one cycle vs. roughly five to ten. On the hot `grid_cutoff!`
periodic path for the user's profile (24 M `mod`s per `msm_energy`
call), even saving five cycles each is ≈ 120 M cycles ≈ 50 ms on a
2.4 GHz core — i.e. an additional ≈ 2× on top of what we already have
for the *periodic* case.

For *open* BC there is no `mod` in the hot path (the bounds-clip happens
at the loop limits), so this optimisation has no effect there. The
current profiling script is open BC; switch to a periodic test to see
the win.

### What to change

The infrastructure already enforces `n_α` divisible by `2^{L-1}` for
periodic axes (via the `2^{L-1}` round-up in `build_grid_hierarchy`).
Adding "and also a power of two" is one extra divisibility check.

Two implementation options, in order of preference:

1. **Detect at codegen.** In `wrap_index` and `_convolve!`, when
   generating the per-axis expression, check `ispow2(Sz[α])` and emit
   `(idx + Sz[α]) & ($(Sz[α]-1))` or similar instead of `mod(idx, Sz[α])`.
   This is local to the existing `@generated` blocks and doesn't change
   any public interface. The `(idx + Sz[α])` is to handle negative
   `idx` correctly without an extra branch (since `Sz[α] > smax_α` we
   have `idx + Sz[α] ≥ 0`).

2. **Force power-of-two grid sizes globally.** Change
   `build_grid_hierarchy` to round `n_α` up to the next power of two
   (instead of the next multiple of `2^{L-1}`). Memory cost is at most
   2× per axis on open BCs; for periodic BCs the cell determines
   `n_α` so the user would need to choose a power-of-2 cell length, but
   most practical periodic cells satisfy this anyway.

Option 1 is the right starting point — local, zero API change, only
fires when the size is already a power of two. We can revisit option
2 if there's pressure to force the case.

### Test we'd want

A periodic-BC version of `profile/1.jl` (currently the script uses
`periodic = ntuple(_ -> false, D)`, switch to `true`) plus a
`PkgBenchmark.judge` comparison between branches. Expected wins
concentrated on `grid_cutoff!_level1` and on any operator that calls
`wrap_index` on a periodic axis. Expected null result on open-BC
benchmarks.
