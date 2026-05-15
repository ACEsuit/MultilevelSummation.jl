# Priorities

Live "what's next" list for `MultilevelSummation.jl`. Tasks are grouped
into three tiers by priority; the recommended order of attack is at the
bottom. Open GitHub issues are referenced by number where relevant.

For the package's design contract (scope, API, interfaces, repository
layout) see [`PLAN.md`](PLAN.md); for the published implementation, the
source itself.

## Tier 1 — High priority (unblocks expansion of the package)

### T1. KernelAbstractions (KA) parallel path — **DONE**

Landed as an *alternative* interface rather than a migration: the
OhMyThreads CPU path is unchanged; a KA path lives alongside in
`src/*_ka.jl`. It is selected per call from the input array type
(`AbstractGPUArray`) or via a `backend = <KA backend>` kwarg on
`msm_energy` / `msm_energy_forces`. Every hot operator has a KA
counterpart (`anterpolate_ka!`, `interpolate_ka!`,
`interpolate_grad_ka!`, `restrict_ka!`, `prolong_ka!`,
`grid_cutoff_ka!`, `top_level_ka!`), and the short-range pair sum
goes through
[NeighbourLists.jl](https://github.com/JuliaMolSim/NeighbourLists.jl)'s
GPU-friendly `SortedCellList` + a per-atom `@kernel`. `MSMCalculator`
is untouched — no `backend` field, no signature change. KA-on-CPU
correctness is exercised against the OhMyThreads path in
`test/test_ka_*.jl`.

**Follow-ups still open:**

- *Short-range pair-count discrepancy in clustered systems.* On H₂O
  with the TIP3P builder, the KA short-range loop (NL.jl
  `SortedCellList` + `for_each_neighbour`) reports 66 ordered pairs
  where the legacy O(N²) loop reports 74 — a ~10 % miss concentrated
  on intramolecular OH partners near the cutoff. The corresponding
  per-atom forces disagree by up to ~50 % on the affected atoms; the
  total energy disagrees by ~1 %. The miss is independent of the GPU
  backend (CPU-KA and GPU-KA agree bit-exactly), so this is a
  cell-list / kernel-side issue rather than a GPU port issue. Already
  marked `@test_broken` in `test/gpu/equivalence.jl`. Plausible root
  causes: strict `<` vs `≤` on the cutoff comparison, or a `for_each_neighbour`
  bug at cell-edge atom placements.
- *GPU-backend CI runner.* The standalone equivalence-check script at
  [`test/gpu/runtests.jl`](test/gpu/runtests.jl) auto-detects whichever
  of `CUDA` / `AMDGPU` / `Metal` / `oneAPI` is installed in
  `test/gpu/`, runs three-way comparison (legacy CPU, KA-on-CPU,
  KA-on-GPU) on NaCl / H2O fixtures, and exits cleanly with a help
  message if none is found. Wiring a self-hosted GitHub Actions runner
  with a GPU into the CI matrix is the missing piece. *Verified
  locally on NVIDIA A100 40GB with CUDA.jl: 10/11 tests pass,
  1 documented `@test_broken` (the H2O force-vs-legacy comparison
  noted above).*
- *AbstractGPUArray-dispatch coverage in standard `]test`.* The
  natural emulator for this is
  [JLArrays.jl](https://github.com/JuliaGPU/JLArrays.jl), but the full
  KA path currently breaks on `JLBackend` upstream: NL.jl's
  `_build_sorted_celllist` ends up in
  AcceleratedKernels.jl's `__forindices_global!`, which has no
  `JLBackend` method, and KA / JLArrays itself doesn't define
  `synchronize(::JLBackend)`. Revisit once those gaps land upstream
  (or once `POCLBackend` is a viable substitute).

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

T1 has landed, so this is now actionable.

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
[`README.md`](README.md) and any stale status text once T6 lands.
Specifically:

- "no ChainRules integration yet" — remove when T6 ships.

The "missing GPU port" caveat is already gone (T1 shipped).

- **Effort**: 30 minutes.
- **Why low**: doc hygiene. Bundle into whichever Tier-2 PR happens
  to touch related sections.

## Recommended sequencing

1. **T2** — splittings for non-Coulomb kernels. Unblocks the
   `InversePower{N≠1}` and `RationalDecay` families that are already
   declared in the API. No collision with T1 (now done): the KA path
   doesn't touch the constructor.
2. **T4** — bench revisit, now that T1 is in.
3. **T5 / T6 / T7** — pick based on the next concrete consumer
   (slab-geometry user, AD user, multi-charge-component user).
4. **T8 / T9 / T10** — opportunistic; T8 is largely subsumed by T1.
