# MultilevelSummation.jl — design contract

A Julia **package** implementation of the **Multilevel Summation Method
(MSM)** following Hardy, Wu, Phillips, Stone, Skeel, Schulten
(*J. Chem. Theory Comput.* 2015, 11, 766–779), generalised to arbitrary
translation-invariant pair kernels with possibly vector-valued charges
and matrix-valued kernels.

This document is the **architectural reference** — the method summary,
the scope and out-of-scope decisions, the repository layout, and the
duck-typed interface contracts that user code can rely on. For *what
is being worked on next*, see [`PRIORITIES.md`](PRIORITIES.md).

## Status

The CPU implementation is functionally complete and multi-threaded
(via OhMyThreads), with 13206 tests passing on `julia -t 1` and
`julia -t 4`. End-to-end MSM (`msm_energy`, `msm_energy_forces`),
the AtomsBase / AtomsCalculators wrapper, the Reference submodule
(naive direct sum + 3D Ewald), and the Tune submodule
(hyperparameter sweeps with realistic NaCl/H2O builders) are all
shipped. The current "highly experimental" caveats — no GPU
backend, only the Coulomb splitting, no ChainRules integration —
are tracked as Tier-1/Tier-2 tasks in [`PRIORITIES.md`](PRIORITIES.md).

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

## 1. Scope

**In scope:**
- Pure Julia, multi-threaded CPU. GPU port via `KernelAbstractions.jl`
  is a planned migration (see [`PRIORITIES.md`](PRIORITIES.md) T1).
- Dimensions `d ∈ {1, 2, 3}` from day one; all operators dimension-generic.
- **Floating-point precision is a free type parameter** `T <: AbstractFloat`
  threaded through positions, charges, cutoffs, grid spacings, and all
  outputs. Default `Float64`; verified for `Float32` in CI.
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
- **Splitting**: pluggable, duck-typed. Currently one concrete
  splitting ships — `HardyC2Cubic` matched to Coulomb
  (`InversePower{1}`). Splittings for general `N ≠ 1` and for
  `RationalDecay` are tracked as `PRIORITIES.md` T2.
- **Pluggable neutralising-background** — user selects via the
  calculator's hyperparameters.
- **Pluggable interpolation basis**; cubic `C¹` (paper §2.2) is the
  default.
- Public API is an `AtomsCalculators.AbstractCalculator` consuming
  `AtomsBase.AbstractSystem`; internally `DecoratedParticles.jl`.
- Units stripped at the AtomsBase boundary; **all arithmetic and all
  tests are unit-free**.
- Unit tests at every layer against an `O(N²)` naive reference; test
  systems are small *random* (not hand-curated) configurations.

**Out of scope:**
- FFT-based top level.
- Multiple time stepping, integrators, NAMD interop.
- Anisotropic translation-invariant kernels not expressible as
  scalar-softened tensor templates (the API doesn't preclude them,
  but no concrete instance ships).

---

## 2. Repository layout

```
MultilevelSummation.jl/
├── Project.toml                    # MultilevelSummation.jl package
├── PLAN.md                         # this file — design contract
├── PRIORITIES.md                   # live "what's next" task list
├── README.md                       # short user-facing intro
├── src/
│   ├── MultilevelSummation.jl      # module, includes, exports
│   ├── calculator.jl               # MSMCalculator + AtomsCalculators glue
│   ├── core.jl                     # low-level msm_energy / msm_energy_forces
│   ├── cell_helpers.jl             # _assert_orthorhombic, _image_ranges, _shift
│   ├── kernels/
│   │   ├── inverse_power.jl        # K(r) = 1/|r|^N  (Coulomb is N=1)
│   │   └── rational_decay.jl       # K(r) = 1/(1 + (|r|/r₀)^N)
│   ├── splittings/
│   │   └── hardy_c2cubic.jl        # Hardy γ for Coulomb (paper)
│   ├── basis/
│   │   └── cubic.jl                # paper's C¹ cubic Φ
│   ├── grid.jl                     # d-D grid with per-axis BC + indexing
│   ├── anterp.jl                   # anterpolation / interpolation (eq. 7, 12)
│   ├── transfer.jl                 # restriction / prolongation (eq. 8, 11)
│   ├── gridcutoff.jl               # local stencil convolution (eq. 9)
│   ├── toplevel.jl                 # top-level direct sum (eq. 10)
│   ├── docstrings.jl               # generic-function docstring shells
│   ├── reference/
│   │   ├── Reference.jl            # submodule wrapper
│   │   ├── ewald.jl                # naive 3D Ewald reference (Coulomb, periodic)
│   │   └── naive.jl                # naive O(N²) direct sum (any kernel, any BC)
│   └── tune/
│       ├── Tune.jl                 # submodule wrapper + exports + includes
│       ├── sweep.jl                # SweepResult, sweep, pareto_front, recommend, run_system_sweep, write_csv
│       ├── summary.jl              # print_pareto_per_N / print_recommendations / print_scaling / print_summary
│       └── systems.jl              # build_nacl, build_h2o, TIP3P + Poisson-disk helpers
├── test/
│   ├── runtests.jl
│   ├── test_kernels.jl             # telescoping identity per (kernel, splitting)
│   ├── test_basis.jl
│   ├── test_splittings.jl
│   ├── test_naive.jl               # tests against Reference.naive_*
│   ├── test_ewald.jl               # self-validate Ewald vs Madelung constants
│   ├── test_anterp.jl
│   ├── test_transfer.jl
│   ├── test_gridcutoff.jl
│   ├── test_toplevel.jl
│   ├── test_core.jl                # low-level end-to-end vs naive + Ewald
│   ├── test_calculator.jl          # AtomsBase + AtomsCalculators path
│   ├── test_systems.jl             # Tune.build_nacl / build_h2o invariants
│   └── test_tune.jl                # Tune.sweep, pareto_front, recommend
├── benchmark/                      # PkgBenchmark suite
├── tuning/                         # tune_NaCl.jl, tune_H2O.jl, plotting.jl
├── docs/                           # Documenter site
└── .github/workflows/              # CI.yml (-t 1 + -t 4 matrix) + Documenter.yml
```

Dependencies: `AtomsBase`, `AtomsCalculators`, `DecoratedParticles`,
`StaticArrays`, `SpecialFunctions`, `Unitful`, `OhMyThreads`,
`ChunkSplitters`, `Random`, `LinearAlgebra`, `Printf`. Test-only:
`StableRNGs`, `AtomsBuilder`, `Unitful`, `Test`.

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
    calc::MSMCalculator{T},
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
struct MSMCalculator{T,K,S,B}    # <: AtomsCalculators.AbstractCalculator
    kernel::K
    splitting::S
    basis::B
    a::T
    h::T
    nlevels::Int
    neutralising_background::Bool
    charge_property::Symbol        # default :charge
end

AtomsCalculators.potential_energy(sys, calc::MSMCalculator) =
    msm_energy(calc, _strip(sys, calc)...)

AtomsCalculators.forces(sys, calc::MSMCalculator)        = ...
AtomsCalculators.energy_forces(sys, calc::MSMCalculator) = ...
AtomsCalculators.forces!(F, sys, calc::MSMCalculator)    = ...
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

Provided:

- `HardyC2Cubic{T}` — Hardy γ matched to `Coulomb` (= `InversePower{1}`),
  paper's `C²` cubic γ (just above eq. 13).

Splittings for general `N ≠ 1` and for `RationalDecay` are an open
design problem tracked as [`PRIORITIES.md`](PRIORITIES.md) T2.

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

Provided:

- `CubicC1{T}` — paper's piecewise cubic, support `|ξ| ≤ 2`.

### 4.4 `MSMCalculator` construction

Constructor responsibilities:
- Validate kernel/splitting compatibility.
- Validate `h ≤ a`, `nlevels ≥ 1`.
- For each periodic axis, check the box length is an integer multiple of
  `2^{L−1} h` (paper §2.1 constraint).
- Infer / fix the working type `T`.

---

## What's next

See [`PRIORITIES.md`](PRIORITIES.md) for the active task list (KA
migration, general splittings, Tune cleanup, etc.) and recommended
sequencing.
