"""
    MultilevelSummation.Tune

Hyperparameter exploration plus a small set of realistic test
configurations used by the tuning scripts, the benchmark suite, and the
test suite as common fixtures.

API surface:

- **System builders** — [`build_nacl`](@ref), [`build_h2o`](@ref).
  These are *experimental staging area* for an upstream PR to
  [AtomsBuilder.jl](https://github.com/JuliaMolSim/AtomsBuilder.jl);
  they may move or change shape.
- **Sweep orchestration** — [`sweep`](@ref), [`run_system_sweep`](@ref).
- **Reference energy** — [`ewald_reference`](@ref).
- **Post-processing** — [`pareto_front`](@ref), [`recommend`](@ref).
- **Text summaries** — [`print_pareto_per_N`](@ref),
  [`print_recommendations`](@ref), [`print_scaling`](@ref),
  [`print_summary`](@ref). Pure stdlib, no plotting deps.
"""
module Tune

using Printf
using Random
using LinearAlgebra: norm
using StaticArrays
using ..MultilevelSummation
using ..MultilevelSummation: HardyC2Cubic, CubicC1, MSMCalculator, msm_energy
using ..MultilevelSummation.Reference: ewald_energy

export SweepResult, sweep, pareto_front, recommend, ewald_reference
export build_nacl, build_h2o
export run_system_sweep, write_csv
export print_pareto_per_N, print_recommendations, print_scaling, print_summary

# --- result table ----------------------------------------------------------

"""
    SweepResult{T}(h, a, L, n_grid, energy, rel_err, t_msm)

One row of a [`sweep`](@ref) result. Fields:

- `h::T` — finest grid spacing.
- `a::T` — short-range cutoff.
- `L::Int` — number of MSM levels.
- `n_grid::Int` — `cell_axis_length / h` (assumed isotropic).
- `energy::T` — MSM energy.
- `rel_err::T` — `|energy − reference| / |reference|`.
- `t_msm::Float64` — wall-clock time of the timed call, in seconds.
"""
struct SweepResult{T<:AbstractFloat}
    h::T
    a::T
    L::Int
    n_grid::Int
    energy::T
    rel_err::T
    t_msm::Float64
end

function Base.show(io::IO, r::SweepResult)
    @printf io "SweepResult(h=%.4g, a=%.4g, L=%d, n_grid=%d, energy=%.6e, rel_err=%.3e, t_msm=%.4fs)" r.h r.a r.L r.n_grid r.energy r.rel_err r.t_msm
end

# --- helpers ---------------------------------------------------------------

# Largest L (≤ L_cap) such that 2^(L-1) divides n_grid.
function _max_levels(n_grid::Int; L_cap::Int = 6)
    L = 1
    while L < L_cap && rem(n_grid, 2^L) == 0
        L += 1
    end
    return L
end

# --- main sweep ------------------------------------------------------------

"""
    sweep(positions, charges, cell, periodic;
          reference::Real,
          basis = CubicC1{T}(),
          h_values = (0.5, 1.0, 2.0),
          a_values = (2.0, 4.0, 8.0),
          L_strategy::Symbol = :max,
          warmup::Bool = true) -> Vector{SweepResult{T}}

Iterate `msm_energy(positions, charges, cell, periodic, calc)` over all
valid `(h, a, L)` settings drawn from `h_values × a_values` and a level
range determined by `L_strategy`:

- `:max` — only the largest valid `L` per `h`.
- `:all` — `L = 2 … L_max`.

A combination is skipped if the cell axis isn't an integer multiple of
`h` or if no `L ≥ 2` has `2^{L-1}` dividing `n_grid = cell/h`.

Each combination is JIT-warmed up once (`warmup=true`) before timing.

The cell is assumed cubic (isotropic) for the `n_grid = cell/h` step;
this matches how the rest of the package picks the grid hierarchy.
"""
function sweep(positions::Vector{SVector{D,T}},
               charges::Vector{T},
               cell::SMatrix{D,D,T},
               periodic::NTuple{D,Bool};
               reference::Real,
               basis = CubicC1{T}(),
               h_values = (0.5, 1.0, 2.0),
               a_values = (2.0, 4.0, 8.0),
               L_strategy::Symbol = :max,
               warmup::Bool = true) where {D, T<:AbstractFloat}

    L_strategy in (:max, :all) ||
        throw(ArgumentError("L_strategy must be :max or :all, got $L_strategy"))

    box = cell[1, 1]                       # assumes cubic / isotropic
    results = SweepResult{T}[]
    abs_ref = abs(reference)

    for h in h_values, a in a_values
        ng_f = box / h
        ng   = round(Int, ng_f)
        abs(ng_f - ng) < 1e-9 || continue   # cell not divisible by h
        L_max = _max_levels(ng)
        L_max ≥ 2 || continue
        Ls = L_strategy === :max ? (L_max,) : (2:L_max)
        for L in Ls
            calc = MSMCalculator(HardyC2Cubic(T(a), L), basis, T(h))
            warmup && msm_energy(positions, charges, cell, periodic, calc)
            t = @elapsed U = msm_energy(positions, charges, cell, periodic, calc)
            rel_err = T(abs(U - reference) / max(abs_ref, eps(T)))
            push!(results, SweepResult{T}(T(h), T(a), L, ng, U, rel_err, t))
        end
    end
    return results
end

# --- post-processing -------------------------------------------------------

"""
    pareto_front(results::AbstractVector{SweepResult{T}}) -> Vector{SweepResult{T}}

Return the subset of `results` that is Pareto-optimal on the
`(rel_err, t_msm)` plane, sorted by ascending `rel_err`.

An entry `r` is dominated if there exists another entry `r'` with both
`rel_err(r') ≤ rel_err(r)` and `t_msm(r') ≤ t_msm(r)` and at least one
strict.
"""
function pareto_front(results::AbstractVector{SweepResult{T}}) where T
    n = length(results)
    keep = trues(n)
    for i in 1:n, j in 1:n
        i == j && continue
        if results[j].rel_err ≤ results[i].rel_err &&
           results[j].t_msm  ≤ results[i].t_msm &&
           (results[j].rel_err < results[i].rel_err ||
            results[j].t_msm  < results[i].t_msm)
            keep[i] = false
            break
        end
    end
    front = results[keep]
    sort!(front; by = r -> r.rel_err)
    return front
end

"""
    recommend(results::AbstractVector{SweepResult}; max_rel_err::Real) -> SweepResult

Return the fastest entry in `results` whose `rel_err ≤ max_rel_err`.
Throws `ArgumentError` if no entry satisfies the bound.
"""
function recommend(results::AbstractVector{<:SweepResult}; max_rel_err::Real)
    feasible = filter(r -> r.rel_err ≤ max_rel_err, results)
    isempty(feasible) &&
        throw(ArgumentError("no sweep entry has rel_err ≤ $max_rel_err " *
                            "(best: $(minimum(r -> r.rel_err, results)))"))
    return reduce((a, b) -> a.t_msm ≤ b.t_msm ? a : b, feasible)
end

# --- Ewald-reference convenience ------------------------------------------

"""
    ewald_reference(positions, charges, cell; tol = 1e-9) -> Real

Compute a 3D-Ewald reference energy for the given fully-periodic
charge-neutral Coulomb system, with `(α, R_cut, k_cut)` auto-tuned for
the prescribed truncation tolerance `tol`.

Selection rule:

    R_cut = min(box/2 − 0.5, 14)
    α     = √(−log tol) / R_cut
    k_cut = 2 α √(−log tol)

Returns the total Ewald energy.
"""
function ewald_reference(positions::Vector{SVector{3,T}},
                         charges::Vector{T},
                         cell::SMatrix{3,3,T};
                         tol::Real = 1e-9) where {T<:AbstractFloat}
    box   = cell[1, 1]                      # assumes cubic
    αR    = sqrt(-log(tol))                  # ≈ 4.55 for tol = 1e-9
    R_cut = min(box / 2 - T(0.5), T(14))
    α     = αR / R_cut
    k_cut = 2 * α * αR
    return ewald_energy(positions, charges, cell; α = α, R_cut = R_cut, k_cut = k_cut)
end

# =============================================================================
# System builders (experimental — staging for an upstream AtomsBuilder PR)
# =============================================================================

"""
    build_nacl(n_super; a_lat = 4.0, σ = 0.0, rng = MersenneTwister(0))
        -> (positions, charges, cell, periodic)

Rock-salt NaCl supercell with charges ±1. `n_super` is the number of
conventional cells per axis (giving `8·n_super³` ions). If `σ > 0`,
every ion gets an independent Gaussian Cartesian displacement of
standard deviation `σ` Å — `σ ≈ 0.1` Å roughly mimics a 300 K thermal
sample. `σ = 0` recovers the perfect lattice exactly.

!!! note "Experimental"
    This builder is a staging area for an upcoming PR to
    `AtomsBuilder.jl`'s experimental submodule. Signature and return
    shape may change once it lands upstream.
"""
function build_nacl(n_super::Int;
                    a_lat::Float64    = 4.0,
                    σ::Float64        = 0.0,
                    rng::AbstractRNG  = MersenneTwister(0))
    na_frac = (SVector(0.0, 0.0, 0.0), SVector(0.5, 0.5, 0.0),
               SVector(0.5, 0.0, 0.5), SVector(0.0, 0.5, 0.5))
    cl_frac = (SVector(0.5, 0.0, 0.0), SVector(0.0, 0.5, 0.0),
               SVector(0.0, 0.0, 0.5), SVector(0.5, 0.5, 0.5))
    positions = SVector{3,Float64}[]
    charges   = Float64[]
    @inbounds for i in 0:n_super-1, j in 0:n_super-1, k in 0:n_super-1
        offset = SVector(Float64(i), Float64(j), Float64(k)) .* a_lat
        for p in na_frac
            r0 = p .* a_lat .+ offset
            δ  = σ == 0 ? zero(SVector{3,Float64}) :
                          σ .* SVector(randn(rng), randn(rng), randn(rng))
            push!(positions, r0 .+ δ)
            push!(charges, +1.0)
        end
        for p in cl_frac
            r0 = p .* a_lat .+ offset
            δ  = σ == 0 ? zero(SVector{3,Float64}) :
                          σ .* SVector(randn(rng), randn(rng), randn(rng))
            push!(positions, r0 .+ δ)
            push!(charges, -1.0)
        end
    end
    box      = n_super * a_lat
    cell     = SMatrix{3,3,Float64}(box * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    return positions, charges, cell, periodic
end

# --- TIP3P water box -------------------------------------------------------
# TIP3P (Jorgensen et al. 1983): rigid, 3-site, q_O = -0.834 e,
# q_H = +0.417 e, r_OH = 0.9572 Å, ∠HOH = 104.52°.

const _TIP3P_Q_O   = -0.834
const _TIP3P_Q_H   = +0.417
const _TIP3P_R_OH  =  0.9572
const _TIP3P_THETA = 104.52 * π / 180

# Reference H positions in the molecule's local frame (O at origin, C₂ axis ‖ ẑ).
const _TIP3P_H1_LOCAL = SVector{3,Float64}( _TIP3P_R_OH * sin(_TIP3P_THETA / 2), 0.0,  _TIP3P_R_OH * cos(_TIP3P_THETA / 2))
const _TIP3P_H2_LOCAL = SVector{3,Float64}(-_TIP3P_R_OH * sin(_TIP3P_THETA / 2), 0.0,  _TIP3P_R_OH * cos(_TIP3P_THETA / 2))

# Uniform random rotation on SO(3) via the unit-quaternion method.
function _random_rotation_matrix(rng::AbstractRNG)
    q = SVector{4,Float64}(randn(rng), randn(rng), randn(rng), randn(rng))
    q = q / norm(q)
    w, x, y, z = q[1], q[2], q[3], q[4]
    return SMatrix{3,3,Float64}(
        1 - 2*(y*y + z*z),   2*(x*y + z*w),       2*(x*z - y*w),
        2*(x*y - z*w),       1 - 2*(x*x + z*z),   2*(y*z + x*w),
        2*(x*z + y*w),       2*(y*z - x*w),       1 - 2*(x*x + y*y),
    )
end

function _pbc_distance²(a::SVector{3,Float64}, b::SVector{3,Float64}, box::Float64)
    dx = a - b
    dx = SVector{3,Float64}(
        dx[1] - box * round(dx[1] / box),
        dx[2] - box * round(dx[2] / box),
        dx[3] - box * round(dx[3] / box),
    )
    return sum(abs2, dx)
end

function _far_from_all(c, positions, d_min²::Float64, box::Float64)
    @inbounds for p in positions
        _pbc_distance²(c, p, box) < d_min² && return false
    end
    return true
end

# Poisson-disk sampler: Bridson's algorithm with k annulus tries per active
# point, followed by uniform-rejection fallback for holes the active list
# missed. Throws if it can't reach n_mol.
function _poisson_disk_oxygens(box::Float64, n_mol::Int, d_min::Float64,
                               rng::AbstractRNG;
                               k::Int = 50,
                               uniform_fallback_trials::Int = 200_000)
    d_min²    = d_min * d_min
    positions = Vector{SVector{3,Float64}}(); sizehint!(positions, n_mol)
    active    = Int[]                         ; sizehint!(active,    n_mol)

    seed = SVector{3,Float64}(box * rand(rng), box * rand(rng), box * rand(rng))
    push!(positions, seed); push!(active, 1)

    # Phase 1: Bridson.
    while !isempty(active) && length(positions) < n_mol
        idx       = rand(rng, 1:length(active))
        center_id = active[idx]
        center    = positions[center_id]
        placed    = false
        for _ in 1:k
            u   = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
            u   = u / norm(u)
            r   = d_min * (1.0 + rand(rng))             # uniform in [d_min, 2·d_min]
            c   = center + r * u
            c   = SVector{3,Float64}(mod(c[1], box), mod(c[2], box), mod(c[3], box))
            if _far_from_all(c, positions, d_min², box)
                push!(positions, c)
                push!(active, length(positions))
                placed = true
                break
            end
        end
        if !placed
            active[idx] = active[end]
            pop!(active)
        end
    end

    # Phase 2: uniform-rejection fallback.
    trials = 0
    while length(positions) < n_mol && trials < uniform_fallback_trials
        trials += 1
        c = SVector{3,Float64}(box * rand(rng), box * rand(rng), box * rand(rng))
        if _far_from_all(c, positions, d_min², box)
            push!(positions, c)
        end
    end

    length(positions) == n_mol ||
        error("Poisson-disk jammed: placed $(length(positions))/$n_mol oxygens " *
              "(box = $box Å, d_min = $d_min Å, fallback trials = $trials). " *
              "Lower d_min or n_mol.")
    return positions
end

"""
    build_h2o(box; ρ = 0.0334, d_min = 2.7, rng = MersenneTwister(0))
        -> (positions, charges, cell, periodic)

TIP3P-like liquid-water configuration in a cubic box of side `box` (Å).
`n_mol = round(ρ · box³)` oxygens placed by Poisson-disk rejection
sampling (min O–O distance `d_min` Å, minimum-image PBC); each molecule
gets a uniformly-random SO(3) orientation. Atom order per molecule is
`[O, H, H]`.

Standard liquid-water values: `ρ = 0.0334 mol/Å³` ≈ 1 g/cm³,
`d_min = 2.7 Å`.

!!! note "Experimental"
    Same caveat as [`build_nacl`](@ref) — staging for an upstream PR.
"""
function build_h2o(box::Float64;
                   ρ::Float64        = 0.0334,
                   d_min::Float64    = 2.7,
                   rng::AbstractRNG  = MersenneTwister(0))
    n_mol     = round(Int, ρ * box^3)
    O_centres = _poisson_disk_oxygens(box, n_mol, d_min, rng)
    positions = Vector{SVector{3,Float64}}(); sizehint!(positions, 3 * n_mol)
    charges   = Vector{Float64}();             sizehint!(charges,   3 * n_mol)
    for O in O_centres
        R  = _random_rotation_matrix(rng)
        H1 = O + R * _TIP3P_H1_LOCAL
        H2 = O + R * _TIP3P_H2_LOCAL
        push!(positions, O);  push!(charges, _TIP3P_Q_O)
        push!(positions, H1); push!(charges, _TIP3P_Q_H)
        push!(positions, H2); push!(charges, _TIP3P_Q_H)
    end
    cell     = SMatrix{3,3,Float64}(box * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    return positions, charges, cell, periodic
end

# =============================================================================
# Sweep orchestration over a sequence of system sizes
# =============================================================================

"""
    run_system_sweep(builder, sizes; ...) -> Vector{NamedTuple}

Iterate a `builder(size) -> (positions, charges, cell, periodic)`
callable over `sizes`, time the Ewald reference, run [`sweep`](@ref),
print a per-system table, and return a flat row vector suitable for
both CSV-writing and the `print_*` summary helpers.

Keyword arguments:
- `h_values`, `a_values`, `L_strategy` — forwarded to [`sweep`](@ref).
- `ewald_tol` — forwarded to [`ewald_reference`](@ref).
- `size_label` — printed name for the `sizes` parameter (e.g. `"n_super"`,
  `"box"`). Cosmetic only.
- `io` — `IO` stream for the per-system tables. Defaults to `stdout`.

Each returned row is a NamedTuple with fields
`(size, N, box, h, n_grid, a, L, rel_err, t_msm, t_ewald, U_ref)`.
"""
function run_system_sweep(builder, sizes;
                          h_values   = (0.5, 1.0, 2.0),
                          a_values   = (2.0, 4.0, 8.0),
                          L_strategy = :all,
                          ewald_tol  = 1e-9,
                          size_label::AbstractString = "size",
                          io::IO     = stdout)
    rows = NamedTuple[]
    for s in sizes
        positions, charges, cell, periodic = builder(s)
        N   = length(positions)
        box = cell[1, 1]
        @printf io "\n=========  %s = %s, N = %d, box = %.2f Å  =========\n" size_label s N box
        t_ewald = @elapsed U_ref = ewald_reference(positions, charges, cell; tol = ewald_tol)
        @printf io "Ewald reference: U_ref = %.6e (run in %.2fs)\n\n" U_ref t_ewald

        results = sweep(positions, charges, cell, periodic;
                        reference  = U_ref,
                        h_values   = h_values,
                        a_values   = a_values,
                        L_strategy = L_strategy)

        @printf io "%4s %6s %3s %5s   %12s %10s\n" "h" "a" "L" "n_grid" "rel_err" "t_msm (s)"
        @printf io "%4s %6s %3s %5s   %12s %10s\n" "----" "------" "---" "------" "------------" "----------"
        for r in results
            @printf io "%4.2f %6.2f %3d %5d   %12.3e %10.4f\n" r.h r.a r.L r.n_grid r.rel_err r.t_msm
            push!(rows, (size = s, N = N, box = box,
                         h = r.h, n_grid = r.n_grid, a = r.a, L = r.L,
                         rel_err = r.rel_err, t_msm = r.t_msm,
                         t_ewald = t_ewald, U_ref = U_ref))
        end
    end
    return rows
end

# =============================================================================
# CSV writer
# =============================================================================

"""
    write_csv(rows, path)

Write the `Vector{NamedTuple}` produced by [`run_system_sweep`](@ref) to
`path` as a CSV. Header: `size,N,box,h,n_grid,a,L,rel_err,t_msm,t_ewald,U_ref`.
Tuning scripts and benchmarks share this format.
"""
function write_csv(rows::AbstractVector, path::AbstractString)
    open(path, "w") do io
        println(io, "size,N,box,h,n_grid,a,L,rel_err,t_msm,t_ewald,U_ref")
        for r in rows
            @printf io "%s,%d,%.4f,%.4f,%d,%.4f,%d,%.6e,%.6e,%.6e,%.6e\n" string(r.size) r.N r.box r.h r.n_grid r.a r.L r.rel_err r.t_msm r.t_ewald r.U_ref
        end
    end
    return path
end

# =============================================================================
# Text summaries (pure stdlib, no plotting deps)
# =============================================================================

# Build a SweepResult from a row NamedTuple. `energy` isn't stored in rows
# (we keep them small for the CSV), but pareto_front/recommend don't read it.
_row_to_sweepresult(r) = SweepResult{Float64}(r.h, r.a, r.L, r.n_grid, NaN, r.rel_err, r.t_msm)

# Simple log-log slope via least squares. Returns NaN if fewer than two points
# or if any value is non-positive (can't take log).
function _loglog_slope(xs::AbstractVector, ys::AbstractVector)
    length(xs) == length(ys) >= 2 || return NaN
    all(>(0), xs) && all(>(0), ys) || return NaN
    lx = log.(xs); ly = log.(ys)
    mx = sum(lx) / length(lx); my = sum(ly) / length(ly)
    num = sum((lx .- mx) .* (ly .- my))
    den = sum((lx .- mx).^2)
    den > 0 ? num / den : NaN
end

"""
    print_pareto_per_N(rows; io = stdout)

Print, per distinct `N`, the Pareto-optimal entries on
`(rel_err, t_msm)` sorted by ascending `rel_err`.
"""
function print_pareto_per_N(rows; io::IO = stdout)
    Ns = sort(unique(r.N for r in rows))
    for N in Ns
        group = [r for r in rows if r.N == N]
        front = pareto_front([_row_to_sweepresult(r) for r in group])
        @printf io "\n--- N = %5d (%d entries; %d on Pareto front) ---\n" N length(group) length(front)
        @printf io "%5s %6s %4s %6s   %12s %12s\n" "h" "a" "L" "ngrid" "rel_err" "t_msm (s)"
        for r in front
            @printf io "%5.2f %6.2f %4d %6d   %12.3e %12.4f\n" r.h r.a r.L r.n_grid r.rel_err r.t_msm
        end
    end
end

"""
    print_recommendations(rows; thresholds = (1e-2, 1e-3, 1e-4), io = stdout)

For each distinct `N` and each `thresholds` entry, print the fastest
sweep row whose `rel_err` falls below the threshold (or "infeasible"
if none does).
"""
function print_recommendations(rows; thresholds = (1e-2, 1e-3, 1e-4), io::IO = stdout)
    println(io, "\n", repeat("-", 72))
    println(io, "Fastest setting below relative-error threshold (per N)")
    println(io, repeat("-", 72))
    @printf io "%6s  %10s  %5s %6s %4s   %12s %10s\n" "N" "threshold" "h" "a" "L" "rel_err" "t_msm"
    Ns = sort(unique(r.N for r in rows))
    for N in Ns, thr in thresholds
        group = [_row_to_sweepresult(r) for r in rows if r.N == N]
        try
            r = recommend(group; max_rel_err = thr)
            @printf io "%6d  %10.0e  %5.2f %6.2f %4d   %12.3e %10.4f\n" N thr r.h r.a r.L r.rel_err r.t_msm
        catch
            @printf io "%6d  %10.0e  (infeasible)\n" N thr
        end
    end
end

"""
    print_scaling(rows; io = stdout)

Print observed log-log slopes:
- `rel_err` vs `h` at fixed `a` (predicted ≈ +p for cubic basis, `p=3`),
  on the largest `N` in `rows`, using the largest valid `L` per `(h, a)`.
- `rel_err` vs `a` at fixed `h` (predicted ≈ −(p+1) = −4).
- `t_msm` vs `N` at fixed `(h, a, L)` (predicted ≈ +1, i.e. `O(N)`).

These are diagnostic — non-monotonic data or saturated errors will
yield meaningless slopes, which we report as-is.
"""
function print_scaling(rows; io::IO = stdout)
    println(io, "\n", repeat("-", 72))
    println(io, "Error scaling (largest N)")
    println(io, repeat("-", 72))
    println(io, "Predicted (paper): rel_err = O(h^p / a^{p+1}) with p = 3 for cubic basis\n")

    N_top = maximum(r.N for r in rows)
    rs_top = [r for r in rows if r.N == N_top]
    # Pick the largest L per (h, a) for a stable scaling comparison.
    keyed = Dict{Tuple{Float64,Float64}, NamedTuple}()
    for r in rs_top
        k = (r.h, r.a)
        if !haskey(keyed, k) || r.L > keyed[k].L
            keyed[k] = r
        end
    end
    rs_topL = collect(values(keyed))

    println(io, "  rel_err vs h at fixed a (slope ≈ +3 expected):")
    for a in sort(unique(r.a for r in rs_topL))
        sub = sort([r for r in rs_topL if r.a == a]; by = r -> r.h)
        slope = _loglog_slope([r.h for r in sub], [r.rel_err for r in sub])
        if length(sub) ≥ 2
            @printf io "    a = %5.2f Å:  slope = %+5.2f   (h values: %s)\n" a slope join((string(r.h) for r in sub), ", ")
        end
    end

    println(io, "\n  rel_err vs a at fixed h (slope ≈ −4 expected):")
    for h in sort(unique(r.h for r in rs_topL))
        sub = sort([r for r in rs_topL if r.h == h]; by = r -> r.a)
        slope = _loglog_slope([r.a for r in sub], [r.rel_err for r in sub])
        if length(sub) ≥ 2
            @printf io "    h = %5.2f Å:  slope = %+5.2f   (a values: %s)\n" h slope join((string(r.a) for r in sub), ", ")
        end
    end

    println(io, "\n", repeat("-", 72))
    println(io, "Cost scaling: t_msm vs N at fixed (h, a, L) (slope ≈ +1 expected)")
    println(io, repeat("-", 72))

    Ns = sort(unique(r.N for r in rows))
    settings_at_N = Dict(N => Set((r.h, r.a, r.L) for r in rows if r.N == N) for N in Ns)
    common = isempty(Ns) ? Set{Tuple{Float64,Float64,Int}}() :
                           intersect(values(settings_at_N)...)
    if isempty(common)
        println(io, "  (no (h, a, L) appears at every N — skipping)")
        return
    end
    for (h, a, L) in sort(collect(common))
        Ns_pts = Float64[]; ts = Float64[]
        for N in Ns, r in rows
            r.N == N || continue
            r.h == h && r.a == a && r.L == L || continue
            push!(Ns_pts, N); push!(ts, r.t_msm)
        end
        slope = _loglog_slope(Ns_pts, ts)
        if length(Ns_pts) ≥ 2
            @printf io "  (h=%.2f, a=%.2f, L=%d):  slope = %+5.2f   (N: %s, t_msm: %s)\n" h a L slope join((string(Int(x)) for x in Ns_pts), ", ") join((@sprintf("%.4g", t) for t in ts), ", ")
        end
    end
end

"""
    print_summary(rows; io = stdout)

Convenience wrapper that runs [`print_pareto_per_N`](@ref),
[`print_recommendations`](@ref), and [`print_scaling`](@ref) in order.
"""
function print_summary(rows; io::IO = stdout)
    println(io, repeat("=", 72))
    println(io, "Sweep summary — $(length(rows)) rows across N = ",
            join(sort(unique(r.N for r in rows)), ", "))
    println(io, repeat("=", 72))
    print_pareto_per_N(rows; io = io)
    print_recommendations(rows; io = io)
    print_scaling(rows; io = io)
end

end # module Tune
