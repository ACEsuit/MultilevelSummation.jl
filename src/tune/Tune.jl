"""
    MultilevelSummation.Tune

Programmatic hyperparameter exploration for MSM. Given a system
(positions, charges, cell, periodic) and an absolute reference energy,
[`sweep`](@ref) iterates over `(h, a, L)` combinations and returns one
[`SweepResult`](@ref) per valid setting; the caller picks what they want
from the table.

Convenience helpers:
- [`ewald_reference`](@ref) — auto-tune Ewald parameters and call the
  3D Ewald reference for a charge-neutral periodic Coulomb system.
- [`pareto_front`](@ref) — keep only entries that are Pareto-optimal on
  the (rel_err, t_msm) plane.
- [`recommend`](@ref) — fastest entry whose relative error is below a
  user-supplied threshold.
"""
module Tune

using Printf
using StaticArrays
using ..MultilevelSummation
using ..MultilevelSummation: HardyC2Cubic, CubicC1, MSMCalculator, msm_energy
using ..MultilevelSummation.Reference: ewald_energy

export SweepResult, sweep, pareto_front, recommend, ewald_reference

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

Selection rule (matches what `tune_NaCl.jl` and similar scripts have
been using):

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

end # module Tune
