using BenchmarkTools
using MultilevelSummation
using MultilevelSummation.Tune: build_nacl, build_h2o
using StaticArrays
using StableRNGs
using Random

# `SUITE` is the entry point PkgBenchmark.jl looks for. Run via:
#
#   julia --project=benchmark -e 'using PkgBenchmark; benchmarkpkg("MultilevelSummation")'
#
# System-scale benchmarks (end_to_end, scaling_N, precision) use the
# realistic NaCl/H2O configurations exposed by `MultilevelSummation.Tune`,
# at the (h, a, L) settings the tuning sweeps identified as
# Pareto-optimal for those systems:
#
#   - NaCl: (h = 2 Å, a = 4 Å, L = 2)
#   - H2O : (h = 2 Å, a = 8 Å, L = 3)
#
# Per-operator and wrap_mode microbenchmarks still drive grids directly
# with `StableRNG`-seeded charges, since their purpose is to track
# operator-level codegen rather than system-scale behaviour.
#
# All defaults are CI-friendly (≲ a few seconds total). PERF_NOTES.md
# tracks history; note that the system-level baselines are now realistic
# fixtures, so judge comparisons against pre-refactor commits are not
# meaningful.

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------
# NaCl + H2O fixture helpers (cached once per fixture)
# ----------------------------------------------------------------------

# σ ≈ 0.1 Å mimics a 300 K thermal sample (see Tune.build_nacl). Pin the
# RNG so the system is stable across benchmark runs.
_nacl(n_super) = build_nacl(n_super; σ = 0.1, rng = MersenneTwister(0xBEEF))
_h2o(box)      = build_h2o(box;      rng = MersenneTwister(0xBEEF))

# Pareto-optimal hyperparameters per system family.
const NACL_CALC = MSMCalculator(HardyC2Cubic(4.0, 2), CubicC1{Float64}(), 2.0)
const H2O_CALC  = MSMCalculator(HardyC2Cubic(8.0, 3), CubicC1{Float64}(), 2.0)

# ----------------------------------------------------------------------
# End-to-end (the headline numbers)
# ----------------------------------------------------------------------
SUITE["end_to_end"] = BenchmarkGroup()

let
    pos, q, cell, per = _nacl(2)                     # 64 ions
    SUITE["end_to_end"]["msm_energy_nacl_n=2"] =
        @benchmarkable msm_energy($pos, $q, $cell, $per, $NACL_CALC)
    SUITE["end_to_end"]["msm_energy_forces_nacl_n=2"] =
        @benchmarkable msm_energy_forces($pos, $q, $cell, $per, $NACL_CALC)

    # Naive O(N²) open-BC reference, same N, for context.
    cell_open = zero(SMatrix{3,3,Float64})
    per_open  = (false, false, false)
    K = Coulomb()
    SUITE["end_to_end"]["naive_energy_open_nacl_n=2"] =
        @benchmarkable naive_energy($pos, $q, $cell_open, $per_open, $K)
end

let
    pos, q, cell, per = _h2o(8.0)                    # ~51 sites
    SUITE["end_to_end"]["msm_energy_h2o_box=8"] =
        @benchmarkable msm_energy($pos, $q, $cell, $per, $H2O_CALC)
    SUITE["end_to_end"]["msm_energy_forces_h2o_box=8"] =
        @benchmarkable msm_energy_forces($pos, $q, $cell, $per, $H2O_CALC)
end

# ----------------------------------------------------------------------
# Per-operator microbenchmarks (grid-driven, unchanged)
# ----------------------------------------------------------------------
SUITE["operators"] = BenchmarkGroup()

let
    D = 3
    h = 0.5
    n_fine = 8
    grid = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n_fine, D),
                       ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
    coarse = coarser_grid(grid, 2)
    basis = CubicC1()

    rng = StableRNG(UInt64(0xBEEF))
    N   = 32
    cell_L = h * n_fine
    positions = [SVector{D,Float64}(ntuple(_ -> rand(rng) * cell_L, Val(D))) for _ in 1:N]
    charges   = randn(rng, Float64, N)
    charges  .-= sum(charges) / N

    q_fine   = zeros(grid.size...)
    q_coarse = zeros(coarse.size...)
    e_fine   = zeros(grid.size...)
    e_coarse = zeros(coarse.size...)
    pots     = zeros(length(positions))
    grads    = zeros(SVector{D,Float64}, length(positions))
    splitting = HardyC2Cubic(2.0, 4)

    SUITE["operators"]["anterpolate!"] =
        @benchmarkable anterpolate!($q_fine, $positions, $charges, $grid, $basis)
    SUITE["operators"]["restrict!"] =
        @benchmarkable restrict!($q_coarse, $q_fine, $coarse, $grid, $basis)
    SUITE["operators"]["prolong!"] =
        @benchmarkable prolong!($e_fine, $e_coarse, $grid, $coarse, $basis)
    SUITE["operators"]["interpolate!"] =
        @benchmarkable interpolate!($pots, $positions, $e_fine, $grid, $basis)
    SUITE["operators"]["interpolate_grad!"] =
        @benchmarkable interpolate_grad!($grads, $positions, $e_fine, $grid, $basis)
    SUITE["operators"]["grid_cutoff!_level1"] =
        @benchmarkable grid_cutoff!($e_fine, $q_fine, $grid, $splitting, 1)
    SUITE["operators"]["top_level!"] =
        @benchmarkable top_level!($e_coarse, $q_coarse, $coarse, $splitting)
end

# ----------------------------------------------------------------------
# Cost scaling vs N: NaCl supercells at the recommended (h, a, L)
# ----------------------------------------------------------------------
# n_super = 2, 3, 4 → 64, 216, 512 ions. The (h=2, a=4, L=2) calc
# above means ngrid scales with box (4, 6, 8) — keeps the L = 2
# hierarchy valid for all three sizes.
SUITE["scaling_N"] = BenchmarkGroup()

for n_super in (2, 3, 4)
    pos, q, cell, per = _nacl(n_super)
    N = length(pos)
    SUITE["scaling_N"]["nacl_n=$n_super (N=$N)"] =
        @benchmarkable msm_energy($pos, $q, $cell, $per, $NACL_CALC)
end

# H2O at box = 8, 16 (51 and 411 sites). box = 12 (ngrid = 6, max L = 2)
# can't host L = 3 so we skip it for an apples-to-apples comparison.
for box in (8.0, 16.0)
    pos, q, cell, per = _h2o(box)
    N = length(pos)
    SUITE["scaling_N"]["h2o_box=$box (N=$N)"] =
        @benchmarkable msm_energy($pos, $q, $cell, $per, $H2O_CALC)
end

# ----------------------------------------------------------------------
# Periodic wrap mode: pow2 (bitmask) vs non-pow2 (mod-with-const)
# ----------------------------------------------------------------------
# Drives `grid_cutoff!` and full `msm_energy` directly on grids of
# pow2 (n=16) and non-pow2 (n=12) extents to keep the wrap-fast-path
# regression visible. Independent of system fixtures.
SUITE["wrap_mode"] = BenchmarkGroup()

let
    D = 3
    h = 0.5
    a = 1.0
    L = 3
    splitting = HardyC2Cubic(a, L)
    basis     = CubicC1()
    calc      = MSMCalculator(splitting, basis, h)

    for (label, n_fine) in (("pow2_n=16", 16), ("nonpow2_n=12", 12))
        rng = StableRNG(UInt64(0xBEEF))
        N   = 32
        cell_L = h * n_fine
        cell   = SMatrix{D,D,Float64}(cell_L * one(SMatrix{D,D,Float64}))
        periodic  = ntuple(_ -> true, Val(D))
        positions = [SVector{D,Float64}(ntuple(_ -> rand(rng) * cell_L, Val(D))) for _ in 1:N]
        charges   = randn(rng, Float64, N)
        charges  .-= sum(charges) / N

        SUITE["wrap_mode"]["msm_energy_$label"] =
            @benchmarkable msm_energy($positions, $charges, $cell, $periodic, $calc)

        grid = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n_fine, D),
                           ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
        q_fine = randn(rng, grid.size...)
        e_fine = zeros(grid.size...)
        SUITE["wrap_mode"]["grid_cutoff!_$label"] =
            @benchmarkable grid_cutoff!($e_fine, $q_fine, $grid, $splitting, 1)
    end
end

# ----------------------------------------------------------------------
# Precision: Float64 vs Float32 cost (NaCl-64 at recommended calc)
# ----------------------------------------------------------------------
SUITE["precision"] = BenchmarkGroup()

let
    for T in (Float64, Float32)
        pos64, q64, cell64, per = _nacl(2)
        pos  = SVector{3,T}.(pos64)
        q    = T.(q64)
        cell = SMatrix{3,3,T}(cell64)
        calc = MSMCalculator(HardyC2Cubic(T(4.0), 2), CubicC1{T}(), T(2.0))
        SUITE["precision"]["msm_energy_$(T)"] =
            @benchmarkable msm_energy($pos, $q, $cell, $per, $calc)
    end
end

# ----------------------------------------------------------------------
# Deep hierarchy: same realistic fixtures, but finer h that pushes
# n_grid up to 16 and allows L = 4. Tracks the multi-level
# restrict/prolong/per-level-gridcutoff code paths that the headline
# (h=2, a≤8) recommendations skip.
# ----------------------------------------------------------------------
SUITE["deep_hierarchy"] = BenchmarkGroup()

let
    calc_deep = MSMCalculator(HardyC2Cubic(2.0, 4), CubicC1{Float64}(), 0.5)
    pos_nacl, q_nacl, cell_nacl, per_nacl = _nacl(2)
    SUITE["deep_hierarchy"]["msm_energy_nacl_n=2_L=4"] =
        @benchmarkable msm_energy($pos_nacl, $q_nacl, $cell_nacl, $per_nacl, $calc_deep)
    pos_h2o, q_h2o, cell_h2o, per_h2o = _h2o(8.0)
    SUITE["deep_hierarchy"]["msm_energy_h2o_box=8_L=4"] =
        @benchmarkable msm_energy($pos_h2o, $q_h2o, $cell_h2o, $per_h2o, $calc_deep)
end
