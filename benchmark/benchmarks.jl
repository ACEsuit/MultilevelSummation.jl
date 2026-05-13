using BenchmarkTools
using MultilevelSummation
using StaticArrays
using StableRNGs

# `SUITE` is the entry point PkgBenchmark.jl looks for. Run via:
#
#   julia --project=benchmark -e 'using PkgBenchmark; benchmarkpkg("MultilevelSummation")'
#
# or interactively:
#
#   using PkgBenchmark
#   results = benchmarkpkg("MultilevelSummation")
#   export_markdown("benchmark.md", results)
#
# Each group below targets a specific MSM operator or the end-to-end
# pipeline. The defaults are chosen modest so the suite is CI-friendly
# (≲ a few seconds total on a laptop). Local profiling: bump `N` and
# `n_fine` in the suite below.

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------
# Helpers — keep parameters in one place so it's easy to scale up later
# ----------------------------------------------------------------------

const D_DEFAULT  = 3
const N_DEFAULT  = 32
const h_DEFAULT  = 0.5
const a_DEFAULT  = 2.0
const L_DEFAULT  = 4
const RNG_SEED   = UInt64(0xBEEF)

# Build a fully-periodic random neutral system in a D-dimensional cube.
function _make_periodic_system(::Val{D}, N::Int, h::T, n_fine::Int) where {D, T<:AbstractFloat}
    rng = StableRNG(RNG_SEED)
    cell_L = h * n_fine
    cell   = SMatrix{D,D,T}(cell_L * one(SMatrix{D,D,T}))
    periodic = ntuple(_ -> true, Val(D))
    positions = [SVector{D,T}(ntuple(_ -> rand(rng) * cell_L, Val(D))) for _ in 1:N]
    charges   = randn(rng, T, N)
    charges  .-= sum(charges) / N
    return positions, charges, cell, periodic
end

function _default_calc(::Type{T}) where {T<:AbstractFloat}
    splitting = HardyC2Cubic(T(a_DEFAULT), L_DEFAULT)
    basis     = CubicC1{T}()
    return MSMCalculator(splitting, basis, T(h_DEFAULT))
end

# ----------------------------------------------------------------------
# End-to-end benchmarks (the headline numbers)
# ----------------------------------------------------------------------
SUITE["end_to_end"] = BenchmarkGroup()

for D in (1, 2, 3)
    positions, charges, cell, periodic = _make_periodic_system(Val(D), N_DEFAULT, h_DEFAULT, 8)
    calc = _default_calc(Float64)
    SUITE["end_to_end"]["msm_energy_D=$D"] =
        @benchmarkable msm_energy($positions, $charges, $cell, $periodic, $calc)
    SUITE["end_to_end"]["msm_energy_forces_D=$D"] =
        @benchmarkable msm_energy_forces($positions, $charges, $cell, $periodic, $calc)
end

# Naive reference for size comparison (Coulomb open BC)
let
    positions, charges, _, _ =
        _make_periodic_system(Val(3), N_DEFAULT, h_DEFAULT, 8)
    cell     = zero(SMatrix{3,3,Float64})
    periodic = (false, false, false)
    K = Coulomb()
    SUITE["end_to_end"]["naive_energy_open_N=$(N_DEFAULT)"] =
        @benchmarkable naive_energy($positions, $charges, $cell, $periodic, $K)
end

# ----------------------------------------------------------------------
# Per-operator microbenchmarks
# ----------------------------------------------------------------------
SUITE["operators"] = BenchmarkGroup()

let
    D = 3
    h = h_DEFAULT
    n_fine = 8
    grid = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n_fine, D),
                       ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
    coarse = coarser_grid(grid, 2)
    basis = CubicC1()

    positions, charges, _, _ = _make_periodic_system(Val(D), N_DEFAULT, h, n_fine)
    q_fine   = zeros(grid.size...)
    q_coarse = zeros(coarse.size...)
    e_fine   = zeros(grid.size...)
    e_coarse = zeros(coarse.size...)
    pots     = zeros(length(positions))
    grads    = zeros(SVector{D,Float64}, length(positions))
    splitting = HardyC2Cubic(a_DEFAULT, L_DEFAULT)

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
# Scaling: end-to-end vs N (open BC Coulomb, single dimension)
# ----------------------------------------------------------------------
SUITE["scaling_N"] = BenchmarkGroup()

for N in (16, 32, 64)
    positions, charges, _, _ = _make_periodic_system(Val(3), N, h_DEFAULT, 8)
    cell     = SMatrix{3,3,Float64}(8 * h_DEFAULT * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    calc = _default_calc(Float64)
    SUITE["scaling_N"]["msm_N=$N"] =
        @benchmarkable msm_energy($positions, $charges, $cell, $periodic, $calc)
    K = Coulomb()
    SUITE["scaling_N"]["naive_periodic_N=$N"] =
        @benchmarkable naive_energy($positions, $charges, $cell, $periodic, $K;
                                     R_cut = 3.0)
end

# ----------------------------------------------------------------------
# Periodic wrap mode: pow2 (bitmask) vs non-pow2 (mod-with-const)
# ----------------------------------------------------------------------
# `UniformGrid{D,T,Per,Sz}` has Sz as a type parameter, so for periodic
# axes whose extent is a power of two the generator emits
# `((idx + n) & (n - 1))`; otherwise it falls back to
# `mod(idx, n)` with `n` an integer literal (multiply-high lowering).
# Both sizes here satisfy `n_fine % 2^(L-1) == 0` (L=3 → mult of 4).
# Smaller `a` keeps the stencil small (smax=ceil(2a/h)) so the suite
# stays CI-friendly.
SUITE["wrap_mode"] = BenchmarkGroup()

let
    D = 3
    h = h_DEFAULT
    a = 1.0
    L = 3
    splitting = HardyC2Cubic(a, L)
    basis     = CubicC1()
    calc      = MSMCalculator(splitting, basis, h)

    for (label, n_fine) in (("pow2_n=16", 16), ("nonpow2_n=12", 12))
        positions, charges, cell, periodic =
            _make_periodic_system(Val(D), N_DEFAULT, h, n_fine)
        SUITE["wrap_mode"]["msm_energy_$label"] =
            @benchmarkable msm_energy($positions, $charges, $cell, $periodic, $calc)

        # Standalone grid_cutoff! at level 1 — the place where the
        # `mod` / bitmask cost concentrates.
        grid = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n_fine, D),
                           ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
        rng = StableRNG(RNG_SEED)
        q_fine = randn(rng, grid.size...)
        e_fine = zeros(grid.size...)
        SUITE["wrap_mode"]["grid_cutoff!_$label"] =
            @benchmarkable grid_cutoff!($e_fine, $q_fine, $grid, $splitting, 1)
    end
end

# ----------------------------------------------------------------------
# Precision: Float64 vs Float32 cost
# ----------------------------------------------------------------------
SUITE["precision"] = BenchmarkGroup()

let
    for T in (Float64, Float32)
        positions, charges, cell, periodic =
            _make_periodic_system(Val(3), N_DEFAULT, T(h_DEFAULT), 8)
        calc = _default_calc(T)
        SUITE["precision"]["msm_energy_$(T)"] =
            @benchmarkable msm_energy($positions, $charges, $cell, $periodic, $calc)
    end
end
