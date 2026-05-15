# Text-summary helpers for sweep results. Pure stdlib + Printf, no
# plotting deps. Operates on the `Vector{NamedTuple}` row format
# produced by `run_system_sweep` and `write_csv` reads.

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
