# Priorities

Live "what's next" list for `MultilevelSummation.jl`. Tasks are grouped
into three tiers by priority; the recommended order of attack is at the
bottom. Open GitHub issues are referenced by number where relevant.

For the package's design contract (scope, API, interfaces, repository
layout) see [`PLAN.md`](PLAN.md); for the published implementation, the
source itself.

## Tier 1 — High priority (unblocks expansion of the package)

### T1. KernelAbstractions (KA) migration

Rewrite each hot operator as a KA `@kernel`, default backend `CPU()`,
optional GPU backends. Replaces the OhMyThreads layer; opens the GPU
port that the README still flags as missing.

- **Files**: every operator (`_convolve!`, `restrict!`, `prolong!`,
  `anterpolate!`, `interpolate!`, `interpolate_grad!`, `top_level!`),
  plus `MSMCalculator` for the optional `backend` field.
- **Effort**: 3–5 days for the CPU path; another 1–2 days for GPU CI
  if a runner is available.
- **Why high**: gates T4 (bench revisit); gates GPU support promised
  in the README; removes the largest "highly experimental" caveat.
  Basic structure: non-generated outer wrapper + `@kernel` inner body,
  `Val{(Per, Sz)}` for per-axis specialisations.

### T2. Splittings for `InversePower{N ≠ 1}` and `RationalDecay`

Design and implement softening / γ functions for the non-Coulomb
kernel families already present in the package. Today,
`MultilevelSummation` only computes for Coulomb because
`HardyC2Cubic` is matched to `K = 1/r` only.

- **Files**: new entries under
  [`src/splittings/`](src/splittings/) (e.g.
  `inverse_power_general.jl`, `rational_decay.jl`). Update the
  `(kernel, splitting)` compatibility check in `MSMCalculator`
  construction.
- **Effort**: 2–4 days including C² continuity proof, FD checks,
  and per-`N` tests. Probably needs at least one design discussion.
- **Why high**: real systems use kernels beyond `1/r`. This was
  PLAN.md's flagship "extends what the prototype can compute" item.

### T3. Issue [#4](https://github.com/ACEsuit/MultilevelSummation.jl/issues/4) — Cleanup `Tune`

1. Split [`src/tune/Tune.jl`](src/tune/Tune.jl) (~500 lines, hard to
   parse) into logical files: `sweep.jl`, `summary.jl` (text
   reporting helpers), `systems.jl` (the experimental `build_nacl` /
   `build_h2o` builders + their helpers), `ewald_reference.jl`. The
   shell `Tune.jl` becomes a module wrapper with `include`s — same
   pattern that landed for `Reference.jl` in the Phase-E1 cleanup.
2. Open the PR to
   [AtomsBuilder.jl](https://github.com/JuliaMolSim/AtomsBuilder.jl)'s
   `Experimental` submodule with `build_nacl` and `build_h2o`. The
   builders' docstrings already announce this move.

- **Effort**: ½ day for the split + ½ day for the upstream PR (plus
  review wait).
- **Why high**: small, isolated, immediate readability win; also
  fulfils a docstring promise.

## Tier 2 — Medium priority (quality / completeness)

### T4. Issue [#6](https://github.com/ACEsuit/MultilevelSummation.jl/issues/6) — Revisit benchmark suite

DEFERRED until T1 lands (issue body says so explicitly).

Replace the small CI-tier fixtures (N=64 NaCl, N=51 H2O) with
informative sizes (n_super=4, box=16); add a quick/full tier toggle;
add a `threading_scaling` group; recompute Pareto-optimal `(h, a, L)`
for the new sizes.

- **Files**: [`benchmark/benchmarks.jl`](benchmark/benchmarks.jl).
- **Effort**: ½ day.
- **Why medium / deferred**: gating on T1 avoids rebaselining twice.

### T5. 2D Ewald / slab-geometry reference

Absolute reference for the mixed-BC case (periodic in `xy`, open in
`z`) — the paper's emphasis. Today's slab tests rely on
translation-invariance + self-consistency, not absolute energy.

- **Files**: new
  [`src/reference/ewald_slab.jl`](src/reference/ewald_slab.jl) (or
  similar); thin wrapper in [`src/reference/Reference.jl`](src/reference/Reference.jl);
  promote relevant tests in
  [`test/test_core.jl`](test/test_core.jl) from invariance-only to
  absolute checks.
- **Effort**: 1–2 days for a sanity-quality implementation
  (Yeh-Berkowitz correction or full slab Ewald). Not a hot path.

### T6. ChainRules integration

AD glue (`rrule` / `frule`) for `msm_energy` and `msm_energy_forces`
so they participate in autodiff pipelines (Zygote, Enzyme on the
front end).

- **Files**: new
  [`src/chainrules.jl`](src/chainrules.jl) as a package extension on
  `ChainRulesCore`.
- **Effort**: 1 day; forces already encode dU/dr so the position
  rrule is almost free. Charges-as-input AD needs extra derivation
  if you want it.
- **Why medium**: README flags "no ChainRules integration yet" as
  one of the three "highly experimental" caveats. AD users hit
  this immediately.

### T7. Vector-charge concrete instance + test

A single small `M = 2` system (e.g. identity-times-Coulomb) to
exercise the `SVector{M,T}` / `SMatrix{M,M,T}` type plumbing that
the codebase already supports in the API surface.

- **Files**: extension in
  [`test/test_core.jl`](test/test_core.jl). No new operator code.
- **Effort**: ½ day.
- **Why medium**: derisks an API claim that's currently untested.

## Tier 3 — Low priority (polish / opportunistic)

### T8. Apply `@generated` to `prolong!` (and revisit `restrict!`)

In the threading round we converted `restrict!` to gather form but
did not promote either it or `prolong!` to `@generated` like
`_convolve!`. Doing so might shave another 10–20% off per-call
cost.

- **Effort**: ½ day per operator if pursued before T1.
- **Why low**: small net wins relative to threading/KA, partially
  overlapping with T1's rewrite.

### T9. Float32 SIMD re-measurement

With the post-threading allocation-free code path, re-bench
`msm_energy_Float32` vs `Float64`. Pre-threading Float32 tied
Float64 because the bottleneck was allocation; now that
allocations are gone, Float32 should pull ahead.

- **Files**: bench code; capture the numbers in a PR description or
  in `docs/src/index.md` (the project no longer maintains a long-lived
  performance log).
- **Effort**: 1 hour.

### T10. Stencil tiling for `grid_cutoff!`

Tile the destination loop in `_convolve!` to fit L1 cache + halo.

- **Effort**: 1–2 days.
- **Why low**: only matters at large grid sizes; T1's GPU path
  may subsume.

### T11. README + PLAN.md status sweep

Refresh the "highly experimental" caveats in
[`README.md`](README.md) and any stale status text once T1 / T6
land. Specifically:

- "missing GPU port via `KernelAbstractions.jl`" — remove when T1
  ships.
- "no ChainRules integration yet" — remove when T6 ships.

- **Effort**: 30 minutes.
- **Why low**: doc hygiene. Bundle into whichever Tier-1/Tier-2
  PR happens to touch related sections.

## Recommended sequencing

1. **T3** — quick, isolated, immediate readability win. Land first.
2. **T11** in passing — fits naturally on the heels of any
   restructuring PR.
3. **T1 and T2 in parallel.** They don't overlap: T1 touches
   operators (`_convolve!`, anter/interpolate, restrict/prolong,
   top_level); T2 touches kernels + splittings (new files in
   `src/splittings/` and a touch to the calculator constructor's
   compatibility check). Two independent PR streams, merged when
   each is ready.
4. **T4** once T1 lands.
5. **T5 / T6 / T7** — pick based on the next concrete consumer
   (slab-geometry user, AD user, multi-charge-component user).
6. **T8 / T9 / T10** — opportunistic; T8 likely superseded by T1.

## Parallelism caveat

If T1 and T2 are pursued by different contributors, watch one
specific merge surface: `MSMCalculator` construction validates
`(kernel, splitting)` compatibility. T2 adds new
`(InversePower{N≠1}, splitting)` pairs; T1 adds a `backend` field.
Both touch the constructor signature. Coordinate the constructor
diff to avoid a painful three-way merge.
