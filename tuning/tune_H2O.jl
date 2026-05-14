# tuning/tune_H2O.jl
#
# Hyperparameter sweep on a TIP3P-like liquid-water box, mirroring the
# structure of `tune_NaCl_perturbed.jl`. Differences vs the NaCl sweeps:
#
#   - Rigid 3-site molecules (one O, two H), not point ions.
#   - Fractional point charges (q_O = -0.834 e, q_H = +0.417 e) with
#     net-zero charge per molecule.
#   - Oxygens placed by Poisson-disk rejection sampling at liquid-water
#     density (≈ 33.3 mol/nm³), with uniformly-random SO(3) orientations
#     per molecule. No grid order, no orientational order.
#
# Output:
#   - Per-box table on stdout.
#   - CSV `tune_H2O_results.csv` next to this script (same column
#     layout as the NaCl scripts so `analyze_NaCl.jl` works unchanged).
#
# Run with:
#
#     cd MultilevelSummation.jl
#     julia --project=tuning tuning/tune_H2O.jl
#
# Edit `BOX_LENGTHS`, `H_VALUES`, `A_VALUES` below to broaden / narrow
# the sweep.

using MultilevelSummation
using MultilevelSummation.Tune: sweep, ewald_reference, SweepResult
using StaticArrays
using Printf
using Random
using LinearAlgebra: norm

# ----------------------------------------------------------------------
# TIP3P constants (Jorgensen et al. 1983)
# ----------------------------------------------------------------------
const Q_O      = -0.834                       # e
const Q_H      = +0.417                       # e
const R_OH     =  0.9572                      # Å
const THETA    = deg2rad(104.52)              # H–O–H angle

# Reference H positions in the molecule's local frame, with O at origin
# and the C₂ axis along +z. Two H sites symmetric about the xz-plane.
const H1_LOCAL = SVector{3,Float64}( R_OH * sin(THETA / 2), 0.0,  R_OH * cos(THETA / 2))
const H2_LOCAL = SVector{3,Float64}(-R_OH * sin(THETA / 2), 0.0,  R_OH * cos(THETA / 2))

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
const BOX_LENGTHS = (16.0, 20.0, 24.0)        # Å
const DENSITY     = 0.0334                    # molecules / Å³ ≈ 1 g/cm³
const D_MIN_OO    = 2.7                       # Å, oxygen exclusion radius
const RNG_SEED    = 42

const H_VALUES = (0.5, 1.0, 2.0)
const A_VALUES = (2.0, 4.0, 8.0)

# ----------------------------------------------------------------------
# Random SO(3) rotation via the quaternion method
# ----------------------------------------------------------------------
"""
    random_rotation_matrix(rng) -> SMatrix{3,3,Float64}

Uniformly random rotation matrix on SO(3): draw a unit quaternion from a
4-D Gaussian (the standard "Shoemake-equivalent" method) and convert to
a 3×3 rotation matrix.
"""
function random_rotation_matrix(rng::AbstractRNG)
    q = SVector{4,Float64}(randn(rng), randn(rng), randn(rng), randn(rng))
    q = q / norm(q)
    w, x, y, z = q[1], q[2], q[3], q[4]
    return SMatrix{3,3,Float64}(
        1 - 2*(y*y + z*z),   2*(x*y + z*w),       2*(x*z - y*w),
        2*(x*y - z*w),       1 - 2*(x*x + z*z),   2*(y*z + x*w),
        2*(x*z + y*w),       2*(y*z - x*w),       1 - 2*(x*x + y*y),
    )
end

# ----------------------------------------------------------------------
# Poisson-disk oxygen placement via Bridson's algorithm (cubic PBC)
# ----------------------------------------------------------------------
# Naive rejection sampling jams well before liquid-water density:
# 137 oxygens in a 16-Å box at d_min = 2.7 Å excludes more total volume
# than the box itself, so the last few placements need exponentially
# many trials. Bridson's algorithm samples each new candidate in a
# narrow spherical shell `[d_min, 2·d_min]` around an existing point,
# which keeps the per-attempt acceptance rate bounded even near jamming.
# See R. Bridson, SIGGRAPH 2007 sketches: "Fast Poisson Disk Sampling
# in Arbitrary Dimensions".

function _pbc_distance²(a::SVector{3,Float64}, b::SVector{3,Float64}, box::Float64)
    dx = a - b
    dx = SVector{3,Float64}(
        dx[1] - box * round(dx[1] / box),
        dx[2] - box * round(dx[2] / box),
        dx[3] - box * round(dx[3] / box),
    )
    return sum(abs2, dx)
end

function _far_from_all(c, positions, d_min², box)
    @inbounds for p in positions
        _pbc_distance²(c, p, box) < d_min² && return false
    end
    return true
end

"""
    sample_oxygens(box, n_mol, d_min, rng; k = 50, uniform_fallback_trials = 200_000)
        -> Vector{SVector{3,Float64}}

Place `n_mol` points in `[0, box]³` (cubic PBC) such that every pair is
at minimum-image distance ≥ `d_min`. Two-phase:

1. **Bridson's annulus sampler** with `k` candidates per active point.
   Fast and gives spatial uniformity, but the active list can empty
   before reaching `n_mol` near jamming density (each active point has
   no annular hole left even though uniform-position holes may exist).
2. **Uniform-rejection fallback** for any remaining slots: draw uniform
   candidates in the box, accept any that respect `d_min` to all
   existing points. Caps at `uniform_fallback_trials`; throws if even
   the fallback can't fill.

Throws on jamming so the caller gets a clean error rather than silent
underfill.
"""
function sample_oxygens(box::Float64, n_mol::Int, d_min::Float64,
                        rng::AbstractRNG;
                        k::Int = 50,
                        uniform_fallback_trials::Int = 200_000)
    d_min² = d_min * d_min
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

    # Phase 2: uniform-rejection fallback for any remaining holes.
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

# ----------------------------------------------------------------------
# Water box assembly
# ----------------------------------------------------------------------
"""
    build_h2o(box; ρ = DENSITY, d_min = D_MIN_OO, rng)

Assemble a TIP3P-like liquid-water configuration:
- `n_mol = round(ρ · box³)` molecules.
- Oxygens by Poisson-disk rejection sampling at minimum O–O distance `d_min`.
- Random uniform SO(3) orientation per molecule.

Returns `(positions, charges, cell, periodic)`. `positions` is
`3·n_mol` long with site order [O, H, H, O, H, H, ...].
"""
function build_h2o(box::Float64;
                   ρ::Float64     = DENSITY,
                   d_min::Float64 = D_MIN_OO,
                   rng::AbstractRNG = MersenneTwister(RNG_SEED))
    n_mol      = round(Int, ρ * box^3)
    O_centres  = sample_oxygens(box, n_mol, d_min, rng)
    positions  = Vector{SVector{3,Float64}}(); sizehint!(positions, 3 * n_mol)
    charges    = Vector{Float64}();             sizehint!(charges,   3 * n_mol)
    for O in O_centres
        R  = random_rotation_matrix(rng)
        H1 = O + R * H1_LOCAL
        H2 = O + R * H2_LOCAL
        push!(positions, O);  push!(charges, Q_O)
        push!(positions, H1); push!(charges, Q_H)
        push!(positions, H2); push!(charges, Q_H)
    end
    cell     = SMatrix{3,3,Float64}(box * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    return positions, charges, cell, periodic, n_mol
end

# ----------------------------------------------------------------------
# Sweep (mirrors run_sweep in tune_NaCl_perturbed.jl)
# ----------------------------------------------------------------------
function run_sweep()
    rows = NamedTuple[]
    for box in BOX_LENGTHS
        positions, charges, cell, periodic, n_mol = build_h2o(box)
        N   = length(positions)
        ρ_actual = n_mol / box^3
        @printf "\n=========  box = %.2f Å, n_mol = %d, N = %d sites, ρ = %.4f /Å³, seed = %d  =========\n" box n_mol N ρ_actual RNG_SEED

        t_ewald = @elapsed U_ref = ewald_reference(positions, charges, cell; tol = 1e-9)
        @printf "Ewald reference: U_ref = %.6e (run in %.2fs)\n\n" U_ref t_ewald

        results::Vector{SweepResult{Float64}} =
            sweep(positions, charges, cell, periodic;
                  reference = U_ref,
                  h_values  = H_VALUES,
                  a_values  = A_VALUES,
                  L_strategy = :all)

        @printf "%4s %6s %3s %5s   %12s %10s\n" "h" "a" "L" "n_grid" "rel_err" "t_msm (s)"
        @printf "%4s %6s %3s %5s   %12s %10s\n" "----" "------" "---" "------" "------------" "----------"
        for r in results
            @printf "%4.2f %6.2f %3d %5d   %12.3e %10.4f\n" r.h r.a r.L r.n_grid r.rel_err r.t_msm
            push!(rows, (N = N, n_super = n_mol, box = box,
                         h = r.h, n_grid = r.n_grid, a = r.a, L = r.L,
                         rel_err = r.rel_err, t_msm = r.t_msm,
                         t_ewald = t_ewald, U_ref = U_ref))
        end
    end
    return rows
end

# ----------------------------------------------------------------------
# Run & dump CSV (column layout matches tune_NaCl[*].jl so the analyser works)
# ----------------------------------------------------------------------
const rows = run_sweep()

csv_path = joinpath(@__DIR__, "tune_H2O_results.csv")
open(csv_path, "w") do io
    println(io, "N,n_super,box,h,n_grid,a,L,rel_err,t_msm,t_ewald,U_ref")
    for r in rows
        @printf io "%d,%d,%.4f,%.4f,%d,%.4f,%d,%.6e,%.6e,%.6e,%.6e\n" r.N r.n_super r.box r.h r.n_grid r.a r.L r.rel_err r.t_msm r.t_ewald r.U_ref
    end
end
@info "wrote $(length(rows)) rows to $csv_path"
