# Sweep, Pareto, recommend, orchestration over multiple system sizes,
# CSV writer. The numerical core of the `Tune` submodule.

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

# --- orchestration over a sequence of system sizes -------------------------

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

# --- CSV writer ------------------------------------------------------------

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
