# MLSum.jl — Prototype Implementation Plan (rev. 5)

A Julia **package** prototype of the **Multilevel Summation Method (MSM)**
following Hardy, Wu, Phillips, Stone, Skeel, Schulten
(*J. Chem. Theory Comput.* 2015, 11, 766–779), generalised to arbitrary
translation-invariant pair kernels with possibly vector-valued charges
and matrix-valued kernels.

Goal: a clean, testable CPU reference that grows into a performant
CPU+GPU implementation via `KernelAbstractions.jl`, exposed as an
`AtomsCalculators.AbstractCalculator`, and ready to register as a Julia
package once correctness is established.

---

## 0. Method summary (generalised)

For positions `r_i ∈ R^D`, charges `q_i ∈ R^M`, and a translation-invariant
pair kernel `K : R^D → R^{M×M}`, MSM approximates

  U ≈ ½ Σ_i Σ_{j ∉ χ(i)}  q_iᵀ K(r_i − r_j) q_j ,    (a scalar)

where `K` is split into one short-range piece `K_0` (compactly supported
in `|r| ≤ a`) and `L` slowly varying pieces `K_1,…,K_L`:

  K = K_0 + K_1 + … + K_L ,

each interpolated on a grid of spacing `h_l = 2^{l−1} h`. The algorithm
has the same shape regardless of `K`:

1. **Anterpolation** of point charges onto the finest grid (eq. 7).
2. **Restriction** to coarser grids (eq. 8).
3. **Local grid-cutoff convolution** on each level with stencil derived
   from `K_l` (eq. 9).
4. **Top-level direct evaluation** at the coarsest grid (eq. 10) —
   optionally with a neutralising background for conditionally convergent
   kernels (Coulomb under full periodicity).
5. **Prolongation** back to finer grids (eq. 11).
6. **Interpolation** to particle positions (eq. 12).

Anter/inter/restrict/prolong are scalar tensor-product operators applied
component-wise to vector charge fields. The grid-cutoff convolution
contracts a matrix-valued stencil with a vector-valued field at each
grid point.

The scalar case `M = 1` (`q_i ∈ R`, `K : R^D → R`) is the paper's setting
and the first concrete instance.

**Standing assumptions.** `K` is translation invariant: `K(r_i, r_j) =
K(r_i − r_j)`. Required by MSM — eq. 9's stencil only depends on
grid-point offsets. Most useful kernels are also isotropic, but the
implementation imposes only translation invariance; isotropy is a
property of specific kernel instances and the splittings designed for
them.

---

## 1. Scope of the first prototype

**In scope (Phase A — CPU prototype):**
- Pure Julia, CPU only, correctness over speed.
- Dimensions `d ∈ {1, 2, 3}` from day one; all operators dimension-generic.
- **Floating-point precision is a free type parameter** `T <: AbstractFloat`
  threaded through positions, charges, cutoffs, grid spacings, and all
  outputs. Default `Float64`; verify a `Float32` run as part of CI.
- **Charge dimension `M`** is a free type parameter, statically known
  (`Q = SVector{M,T}`, kernel value `SMatrix{M,M,T}`). For `M = 1` we
  collapse to scalars (`T`, `T`) for zero overhead.
- Per-axis BC: `:open` / `:periodic` (mixed BCs fall out automatically).
- **Arbitrary translation-invariant kernels** with a duck-typed
  interface. Two concrete *families* shipped:
  - `InversePower{N,T}` — `K(r) = 1/|r|^N`. `N = 1` is Coulomb.
  - `RationalDecay{N,T}` — `K(r) = 1/(1 + (|r|/r₀)^N)`, smooth at the
    origin, `r₀^{-N}` tail.
  No abstract supertype until shared dispatch emerges.
- **Splitting**: pluggable, duck-typed. Only one concrete splitting
  ships in the prototype — `HardyC2Cubic` matched to Coulomb
  (`InversePower{1}`). Splittings for general `N` and for
  `RationalDecay` are a separate design problem deferred until the
  Coulomb path is end-to-end correct.
- **Pluggable neutralising-background** — user selects via the
  calculator's hyperparameters.
- **Pluggable interpolation basis**; cubic `C¹` (paper §2.2) is the
  default. *Note: revisit later.*
- Public API is an `AtomsCalculators.AbstractCalculator` consuming
  `AtomsBase.AbstractSystem`; internally `DecoratedParticles.jl`.
- Units stripped at the AtomsBase boundary; **all arithmetic and all
  tests are unit-free**.
- Unit tests at every layer against an `O(N²)` naive reference; test
  systems are small *random* (not hand-curated) configurations.

**Out of scope for now (Phase B+):**
- FFT-based top level.
- GPU kernels via `KernelAbstractions.jl`. CPU code is written
  "kernel-shaped" so the port is mechanical.
- Multiple time stepping, integrators, NAMD interop.
- Anisotropic translation-invariant kernels not expressible as
  scalar-softened tensor templates (we won't preclude them in the API,
  but no concrete instance ships).

---

## 2. Repository layout

```
MLSum.jl/
├── Project.toml                    # MLSum.jl package
├── PLAN.md                         # this file
├── src/
│   ├── MLSum.jl                    # module, exports
│   ├── calculator.jl               # MLSumCalculator + AtomsCalculators glue
│   ├── api.jl                      # AtomsBase ↔ core: unit stripping, particle build
│   ├── core.jl                     # low-level msm_energy / msm_energy_forces
│   ├── kernels/
│   │   ├── inverse_power.jl        # K(r) = 1/|r|^N  (Coulomb is N=1)
│   │   └── rational_decay.jl       # K(r) = 1/(1 + (|r|/r₀)^N)
│   ├── splittings/
│   │   └── hardy_c2cubic.jl        # Hardy γ for Coulomb (paper)
│   ├── basis/
│   │   └── cubic.jl                # paper's C¹ cubic Φ
│   ├── naive.jl                    # O(N²) reference (open, periodic, mixed)
│   ├── grid.jl                     # d-D grid with per-axis BC + indexing
│   ├── anterp.jl                   # anterpolation / interpolation (eq. 7, 12)
│   ├── transfer.jl                 # restriction / prolongation (eq. 8, 11)
│   ├── gridcutoff.jl               # local stencil convolution (eq. 9)
│   └── toplevel.jl                 # top-level direct sum (eq. 10)
└── test/
    ├── runtests.jl
    ├── refs/
    │   └── ewald.jl                # naive 3D Ewald reference (Coulomb)
    ├── test_kernels.jl             # telescoping identity per (kernel, splitting)
    ├── test_basis.jl
    ├── test_naive.jl
    ├── test_ewald.jl               # self-validate Ewald vs Madelung constants
    ├── test_anterp.jl
    ├── test_transfer.jl
    ├── test_gridcutoff.jl
    ├── test_toplevel.jl
    ├── test_core.jl                # low-level end-to-end vs naive + Ewald
    └── test_calculator.jl          # AtomsBase + AtomsCalculators path
```

Dependencies: `AtomsBase`, `AtomsCalculators`, `DecoratedParticles`,
`StaticArrays`. Test only: `StableRNGs`, `AtomsBuilder`, `Unitful`,
`Test` (stdlib).

---

## 3. Two-layer API: low-level core + AtomsCalculator wrapper

### 3.1 Low-level core (numerical, unit-free)

The numerical entry points take raw arrays — no system object, no units,
no AtomsBase. This is what gets ported to KernelAbstractions later, and
what tests exercise directly.

```julia
# All array element types share the same T <: AbstractFloat.
# Q is either T (scalar charges) or SVector{M,T} (vector charges).

msm_energy(
    calc::MLSumCalculator{T},
    positions::AbstractVector{SVector{D,T}},
    charges::AbstractVector{Q},
    cell::SMatrix{D,D,T},
    periodic::NTuple{D,Bool},
) -> T

msm_energy_forces(
    calc, positions, charges, cell, periodic,
) -> (energy::T, forces::Vector{SVector{D,T}})
```

`cell` is the unit-free `D×D` lattice matrix; rows/columns for periodic
axes are constrained (see §4.4). Non-periodic axes can carry any
non-zero entries on the diagonal (they're not used).

### 3.2 AtomsCalculators wrapper

```julia
struct MLSumCalculator{T,K,S,B}    # <: AtomsCalculators.AbstractCalculator
    kernel::K
    splitting::S
    basis::B
    a::T
    h::T
    nlevels::Int
    neutralising_background::Bool
    charge_property::Symbol        # default :charge
end

AtomsCalculators.potential_energy(sys, calc::MLSumCalculator) =
    msm_energy(calc, _strip(sys, calc)...)

AtomsCalculators.forces(sys, calc::MLSumCalculator)        = ...
AtomsCalculators.energy_forces(sys, calc::MLSumCalculator) = ...
AtomsCalculators.forces!(F, sys, calc::MLSumCalculator)    = ...
```

`_strip` reads `position`, `cell`, `periodicity`, and the configured
charge property from the AtomsBase system, asserts unit consistency
against the kernel's expected scale, strips to `T`, and constructs the
internal `DecoratedParticles` state. The calculator's `T` is inferred
from `a, h` and required to match the stripped position/charge type.

This split means:
- Most of the code is unit-free, AtomsBase-free, calculator-free, and
  trivially testable.
- The AtomsCalculator wrapper is thin: argument extraction + unit
  stripping + dispatch.

---

## 4. Kernel, splitting, basis: duck-typed interfaces

No abstract supertypes for now. Each interface is a list of methods a
user-provided type must implement; we'll introduce supertypes only when
shared code/dispatch demands it.

### 4.1 Kernel

A kernel `K` must implement:

```julia
(K)(r::SVector{D,T})  ->  KV       # KV = T  or  SMatrix{M,M,T}
grad(K, r::SVector{D,T})  ->  SVector{D, KV}
```

Provided families (both scalar, `M = 1`, `KV = T`):

- `InversePower{N,T}` — `K(r) = 1/|r|^N`. `N` is a type parameter
  (compile-time integer) for branch-free `grad`. `Coulomb{T} =
  InversePower{1,T}` is provided as an alias.
- `RationalDecay{N,T}` — `K(r) = 1/(1 + (|r|/r₀)^N)`, with field
  `r₀::T`. Smooth at the origin; useful as a non-singular kernel for
  testing the grid machinery in isolation from Coulomb-specific corner
  cases.

### 4.2 Splitting

A splitting bundles the softening function and decomposition parameters
matched to a chosen kernel. It must implement:

```julia
short_range(s, r::SVector{D,T})        ->  KV   # K_0
long_range_level(s, l::Int, r)         ->  KV   # K_l, l = 1..L-1
top_level(s, r)                        ->  KV   # K_L
# gradient counterparts:
short_range_grad(s, r)                 ->  SVector{D, KV}
long_range_level_grad(s, l, r)         ->  SVector{D, KV}
top_level_grad(s, r)                   ->  SVector{D, KV}
# flag:
requires_neutralising_background(s)    ->  Bool
```

Provided initially:

- `HardyC2Cubic{T}` — Hardy γ matched to `Coulomb` (= `InversePower{1}`),
  paper's `C²` cubic γ (just above eq. 13).

Splittings for general `N ≠ 1` and for `RationalDecay` are an open
design question, deferred until the Coulomb path is green end-to-end
(see §7).

The `(kernel, splitting)` pair is validated at calculator construction;
mismatched pairs throw early.

### 4.3 Interpolation basis

A basis must implement:

```julia
support_radius(B)  ->  Int                   # in grid spacings, e.g. 2 for cubic
eval_phi(B, ξ::T)        ->  T
eval_phi_prime(B, ξ::T)  ->  T
```

`d`-dimensional basis values are built per particle as tensor products
of the 1-D `Φ` (small loop, no Kronecker allocations).

Provided initially:

- `CubicC1{T}` — paper's piecewise cubic, support `|ξ| ≤ 2`.

*Note: revisit basis order (Hermite, quintic, septic) once the rest of
the stack is green.*

### 4.4 `MLSumCalculator` construction

Constructor responsibilities:
- Validate kernel/splitting compatibility.
- Validate `h ≤ a`, `nlevels ≥ 1`.
- For each periodic axis, check the box length is an integer multiple of
  `2^{L−1} h` (paper §2.1 constraint).
- Infer / fix the working type `T`.

---

## 5. Phase-by-phase build, with tests

Each phase ends green before the next begins. **All tests use small
random configurations** (typically `N = 5..50` particles) and compare
against the naive `O(N²)` reference, except for kernel-level identities
that don't need a configuration.

### Phase 1 — Math primitives (≈ 1 day)

**Build:** `InversePower{N,T}` (incl. `Coulomb` alias), `RationalDecay{N,T}`,
`HardyC2Cubic` (for Coulomb only), `CubicC1`.

**Tests:**
- **Telescoping identity** (per kernel + splitting): random `r ∈ R^D`,
  `K_0(r) + Σ K_l(r) + K_L(r) ≈ K(r)` to machine precision for `T =
  Float64` and to `Float32` precision for `T = Float32`.
- **Support of `K_0`:** `K_0(r) = 0` for `|r| ≥ a`.
- **Continuity of γ:** finite-diff check across `R = 1`.
- **Basis partition of unity:** random `x` in support interior.
- **Basis polynomial reproduction:** cubic basis exactly reproduces
  polynomials up to degree 3 (random coefficients).

### Phase 2 — Naive reference (≈ ½ day)

**Build:**
- `naive_energy_forces(positions, charges, cell, periodic, kernel)` —
  generic in `T`, `Q`, kernel, BC.
  - `:open`: direct double sum.
  - `:periodic` / mixed: minimum-image truncated-image sum to a large
    `R_max`; accurate for fast-decaying kernels (`1/r⁶`) by construction.

**Tests:**
- **Random `N = 5` system, open BC, multiple kernels (`Coulomb`,
  `InversePower{6}`, `RationalDecay{4}`), `d ∈ {1,2,3}`:** energy and
  forces match a brute-force double sum implemented independently in
  the test to machine precision — this tests *the naive itself*
  against an independent naive.
- **Random small system, periodic, fast-decaying kernels
  (`InversePower{6}`, `RationalDecay{N}`):** truncated-image sum
  converges geometrically as the truncation radius grows.
- **FD gradient check on the naive forces** (small random system).

### Phase 2b — Ewald reference for 3D periodic Coulomb (≈ ½ day)

Test-only infrastructure. Lives in `test/refs/ewald.jl`, not shipped
with the package.

**Build:**
- `ewald_energy_forces(positions, charges, cell; α, R_cut, k_cut)`:
  the standard 3D Ewald decomposition (real-space erfc sum + reciprocal
  Gaussian sum + self term). Naive `O(N²)` real-space and `O(N²·K)`
  reciprocal; charge-neutral input required.
- Limited to fully 3D-periodic, orthorhombic cells (sufficient for our
  test needs).

**Tests (self-validation, before it can be used as a reference):**
- **Madelung constants:** energy per ion for NaCl-, CsCl-, ZnS-type
  lattices reproduces literature values to ≥ 8 digits when
  `(α, R_cut, k_cut)` are set conservatively.
- **Parameter invariance:** sweeping `α` over a reasonable range (with
  `R_cut, k_cut` scaled accordingly) leaves the energy invariant to
  ≥ 10 digits — Ewald's defining property and a strong correctness
  check.
- **FD gradient check** on a small random neutral system.
- **Consistency with `naive_energy_forces` for `1/r⁶`-only periodic
  systems** — sanity that the periodic geometry handling matches.

### Phase 3 — Grid + anterpolation / interpolation (≈ 1 day)

**Build:** `Grid{D,T}` (per-axis BC, wrapping helpers), `anterpolate!`,
`interpolate!` for vector charges.

**Tests** (all on random configurations):
- **Transpose property:** `⟨x, Anterp y⟩ = ⟨Interp x, y⟩` on random
  vectors.
- **Charge conservation:** total grid charge = total particle charge.
- **Polynomial reproduction** of vector polynomial fields at random
  particle sites (degree ≤ basis order).
- **Scalar vs vector parity:** for `M = 1` vector charges and the same
  data as scalars, outputs agree.

### Phase 4 — Restriction / prolongation (≈ ½ day)

**Build:** `restrict!`, `prolong!`, tensor-product, per-axis wrapping.

**Tests** (random fields):
- Transpose property between restriction and prolongation.
- Polynomial reproduction.
- Smooth-field round-trip preserves smooth modes within tolerance.

### Phase 5 — Grid-cutoff convolution (≈ 1 day)

**Build:** stencil precomputation per level (`SMatrix`-valued entries
when `M > 1`); direct convolution with per-axis wrap/truncate.

**Tests:**
- Single non-zero grid charge ⇒ output equals stencil (random charge
  value, random grid location).
- Linearity on random fields.
- **Random grid charges**: compare against an explicit eq. 9 double-loop
  reference on small grids.

### Phase 6 — Top level (≈ ½ day)

**Build:** open-axis direct sum; periodic-axis collapse to one point;
neutralising background applied when configured.

**Tests** (random configurations):
- Small fully-open system: brute-force `K_L` agrees with implementation.
- Fully periodic Coulomb: top-level grid charge sums to zero.
- Fully periodic `1/r⁶`: no background, top level matches direct sum.

### Phase 7 — Low-level end-to-end (≈ 1 day)

**Build:** `msm_energy`, `msm_energy_forces` wiring eq. 5 / eq. 13 on
the raw-array core.

Since the prototype only ships a Coulomb splitting, MSM end-to-end tests
run on **Coulomb only**. The other kernels are exercised only through
the naive reference in Phase 2.

**Tests** (the headline ones, all random):
- **Convergence sweep** in `h` and `a` on a fixed random `N = 10` open-BC
  Coulomb system: error scales as `O(h^p / a^{p+1})` with `p = 3` for
  cubic basis.
- **Random `N = 20`**, `d ∈ {1,2,3}`, open BC, Coulomb: relative force
  error vs naive below threshold (calibrated from the sweep).
- **Fully 3D-periodic Coulomb** (neutral random system): relative force
  and energy errors vs the **Ewald reference** below threshold.
- **Mixed BC** (periodic in `xy`, open in `z`), Coulomb: no absolute
  reference (2D Ewald deferred); tested via
  - lattice-translation invariance along the periodic axes;
  - FD gradient check matches returned forces;
  - self-consistency under doubling of `h` and `a`.
- **FD gradient check** on all configurations.
- **Translation invariance** (open and lattice-periodic).
- **`Float32` smoke test** — same suite at lower thresholds.

### Phase 8 — AtomsCalculators wrapper (≈ ½ day)

**Build:** `MLSumCalculator`, AtomsCalculators methods, AtomsBase
extraction, unit stripping.

**Tests:**
- `AtomsCalculators.potential_energy(sys, calc) ≈ msm_energy(calc, ...)`
  on a small `AtomsBuilder` system (e.g. `bulk(...)` or a hand-built
  water box) with hand-attached charges.
- `AtomsCalculators.forces` and `forces!` agree with `msm_energy_forces`.
- Unit-stripping round-trips: feeding the same configuration with
  different consistent unit choices yields identical numerical results.
- A unit *mismatch* (e.g. charge in `e` vs `C` without an explicit
  scale) throws.

---

## 6. GPU migration plan (Phase B)

Two principles in the CPU code now so the port is small:

1. **No closures over mutable state in inner loops.** Every kernel-shaped
   routine takes plain arrays + scalars; allocations live outside.
2. **One function per kernel.** `anterpolate!`, `restrict!`,
   `gridcutoff!`, `prolong!`, `interpolate!`, `shortrange!` each are a
   single loop nest with no method dispatch inside.

Steps:
1. Add `KernelAbstractions`. Rewrite each `*!` as `@kernel`; CPU backend
   must reproduce results bit-for-bit on `Float64` and `Float32`.
2. Run full low-level test suite under the KA-CPU backend (the
   AtomsCalculator wrapper need not change).
3. Add GPU CI path (CUDA / Metal) and re-run.
4. *Only then* optimise: stencil tiling for `gridcutoff!`, gather-based
   `anterpolate!`, coalesced `interpolate!`.

---

## 7. Remaining open questions (deferred, not blockers)

1. **Splittings for `InversePower{N ≠ 1}` and `RationalDecay`.** Hardy
   γ isn't directly reusable for these. A generic design for these
   families is the next thing to discuss once the Coulomb path is
   green. Until then, MSM end-to-end only runs on Coulomb.
2. **2D Ewald / slab geometry reference.** Mixed BC (periodic in `xy`,
   open in `z`) — the configuration the paper most emphasises — has no
   absolute reference in the prototype. Phase 7 falls back to
   invariance + self-consistency for that case. Revisit when needed.
3. **AtomsBase charge property name.** Hard-coded to `:charge` for now;
   generalise later.
4. **Test thresholds.** Calibrate empirically from the Phase-7
   convergence sweep.
5. **Vector-charge concrete instance.** Prototype covers `M = 1`
   everywhere; a single trivial `M = 2` test (e.g. identity-times-Coulomb)
   confirms the type plumbing without shipping a "real" vector kernel.
   Extended later.

---

## 8. Rough effort estimate

| Phase | Description                                | Effort |
|-------|--------------------------------------------|--------|
| 1     | Kernels, basis, splitting primitives       | 1 day  |
| 2     | Naive reference (multi-kernel, mixed BC)   | ½ day  |
| 2b    | Ewald reference in `test/refs/`            | ½ day  |
| 3     | Grid + anterp/interp (vector charges)      | 1 day  |
| 4     | Restriction / prolongation                 | ½ day  |
| 5     | Grid-cutoff convolution (matrix stencil)   | 1 day  |
| 6     | Top level (all BCs, neutralising bg)       | ½ day  |
| 7     | Low-level end-to-end                       | 1 day  |
| 8     | AtomsCalculators wrapper + AtomsBase glue  | ½ day  |
| —     | **CPU prototype total**                    | **≈ 6½ days** |
| B     | KernelAbstractions port + GPU CI           | 2–3 days |
| B+    | GPU optimisation                           | open-ended |
