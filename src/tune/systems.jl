# Experimental system builders (NaCl supercell, TIP3P-like water box)
# used by the tuning scripts, the benchmark suite, and the test suite
# as common realistic fixtures.
#
# These are a *staging area* for an upstream PR to AtomsBuilder.jl's
# experimental submodule. Signature and return shape may change once
# they land upstream.

# --- NaCl supercell --------------------------------------------------------

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
