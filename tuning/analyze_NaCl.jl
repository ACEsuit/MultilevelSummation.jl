# tuning/analyze_NaCl.jl
#
# Reads a sweep CSV produced by `tune_NaCl.jl` (or its perturbed sister),
# prints a text summary (Pareto front per N, fastest entries below error
# thresholds, observed scaling exponents), and writes PNG plots into
# `tuning/plots/`.
#
# Run with:
#
#     julia --project=tuning tuning/analyze_NaCl.jl                       # tune_NaCl_results.csv
#     julia --project=tuning tuning/analyze_NaCl.jl path/to/results.csv   # arbitrary CSV
#
# Plot filenames take a prefix derived from the CSV basename
# (stripping the `tune_` and `_results.csv` brackets, e.g.
# `tune_NaCl_perturbed_results.csv` → `nacl_perturbed_*.png`).

using DelimitedFiles
using Printf
using Plots
using MultilevelSummation.Tune: SweepResult, pareto_front, recommend

const HERE = @__DIR__

# CLI: optional CSV path.
const CSV_PATH = isempty(ARGS) ? joinpath(HERE, "tune_NaCl_results.csv") : ARGS[1]

# Plot prefix: e.g. "tune_NaCl_perturbed_results.csv" -> "nacl_perturbed".
function _plot_prefix(path)
    base = replace(basename(path), r"^tune_" => "", r"_results\.csv$" => "")
    return lowercase(base)
end
const PLOT_PREFIX = _plot_prefix(CSV_PATH)
const PLOT_DIR    = joinpath(HERE, "plots")

isdir(PLOT_DIR) || mkdir(PLOT_DIR)

# ----------------------------------------------------------------------
# Load CSV → vector of (N, SweepResult) rows
# ----------------------------------------------------------------------

isfile(CSV_PATH) || error("CSV not found at $CSV_PATH — run tune_NaCl.jl first")

raw, header = readdlm(CSV_PATH, ','; header = true)
header = vec(header)

col(name) = findfirst(==(name), header)
const c_N       = col("N")
const c_h       = col("h")
const c_ngrid   = col("n_grid")
const c_a       = col("a")
const c_L       = col("L")
const c_relerr  = col("rel_err")
const c_tmsm    = col("t_msm")

struct Row
    N::Int
    sr::SweepResult{Float64}
end

rows = Row[]
for r in eachrow(raw)
    sr = SweepResult{Float64}(
        Float64(r[c_h]),
        Float64(r[c_a]),
        Int(r[c_L]),
        Int(r[c_ngrid]),
        NaN,                  # energy column not loaded — not needed for analysis
        Float64(r[c_relerr]),
        Float64(r[c_tmsm]),
    )
    push!(rows, Row(Int(r[c_N]), sr))
end

const N_VALUES = sort(unique(r.N for r in rows))

# rows_for_N(N) -> Vector{SweepResult}
rows_for_N(N) = [r.sr for r in rows if r.N == N]

# ----------------------------------------------------------------------
# Text summary
# ----------------------------------------------------------------------

println(repeat("=", 72))
println("NaCl hyperparameter sweep — analysis")
println(repeat("=", 72))
@printf "Loaded %d rows across N = %s\n\n" length(rows) join(N_VALUES, ", ")

# Pareto front per N
for N in N_VALUES
    rs = rows_for_N(N)
    front = pareto_front(rs)
    @printf "\n--- N = %5d (%d entries; %d on Pareto front) ---\n" N length(rs) length(front)
    @printf "%5s %6s %4s %6s   %12s %12s\n" "h" "a" "L" "ngrid" "rel_err" "t_msm (s)"
    for r in front
        @printf "%5.2f %6.2f %4d %6d   %12.3e %12.4f\n" r.h r.a r.L r.n_grid r.rel_err r.t_msm
    end
end

# Recommended setting at decreasing thresholds
println("\n", repeat("-", 72))
println("Fastest setting below relative-error threshold (per N)")
println(repeat("-", 72))
@printf "%6s  %10s  %5s %6s %4s   %12s %10s\n" "N" "threshold" "h" "a" "L" "rel_err" "t_msm"
for N in N_VALUES, thr in (1e-2, 1e-3, 1e-4)
    rs = rows_for_N(N)
    try
        r = recommend(rs; max_rel_err = thr)
        @printf "%6d  %10.0e  %5.2f %6.2f %4d   %12.3e %10.4f\n" N thr r.h r.a r.L r.rel_err r.t_msm
    catch e
        @printf "%6d  %10.0e  (infeasible)\n" N thr
    end
end

# ----------------------------------------------------------------------
# Scaling: error vs h (fixed a), error vs a (fixed h)
# ----------------------------------------------------------------------

# Simple log-log slope fitter via least squares.
function loglog_slope(xs::Vector{Float64}, ys::Vector{Float64})
    @assert length(xs) == length(ys) ≥ 2
    lx = log.(xs); ly = log.(ys)
    mx = sum(lx) / length(lx); my = sum(ly) / length(ly)
    num = sum((lx .- mx) .* (ly .- my))
    den = sum((lx .- mx).^2)
    return num / den
end

println("\n", repeat("-", 72))
println("Error scaling (largest N)")
println(repeat("-", 72))

N_top = maximum(N_VALUES)
rs_top = rows_for_N(N_top)

# Want only the largest L per (h, a) for consistent scaling.
function largest_L_per_ha(rs)
    keyed = Dict{Tuple{Float64,Float64},SweepResult{Float64}}()
    for r in rs
        k = (r.h, r.a)
        if !haskey(keyed, k) || r.L > keyed[k].L
            keyed[k] = r
        end
    end
    return collect(values(keyed))
end

rs_top_L = largest_L_per_ha(rs_top)

println("Predicted (paper): rel_err = O(h^p / a^{p+1}) with p = 3 for cubic basis\n")

println("  rel_err vs h at fixed a (slope ≈ +3 expected):")
for a in sort(unique(r.a for r in rs_top_L))
    sub = sort([r for r in rs_top_L if r.a == a]; by = r -> r.h)
    if length(sub) ≥ 2
        slope = loglog_slope([r.h for r in sub], [r.rel_err for r in sub])
        @printf "    a = %5.2f Å:  slope = %+5.2f   (h values: %s)\n" a slope join((string(r.h) for r in sub), ", ")
    end
end

println("\n  rel_err vs a at fixed h (slope ≈ −4 expected):")
for h in sort(unique(r.h for r in rs_top_L))
    sub = sort([r for r in rs_top_L if r.h == h]; by = r -> r.a)
    if length(sub) ≥ 2
        slope = loglog_slope([r.a for r in sub], [r.rel_err for r in sub])
        @printf "    h = %5.2f Å:  slope = %+5.2f   (a values: %s)\n" h slope join((string(r.a) for r in sub), ", ")
    end
end

# ----------------------------------------------------------------------
# Cost scaling: t_msm vs N for a fixed (h, a, L)
# ----------------------------------------------------------------------

println("\n", repeat("-", 72))
println("Cost scaling: t_msm vs N at fixed (h, a, L) (slope ≈ +1 expected)")
println(repeat("-", 72))

# Pick (h, a, L) settings that appear at every N
settings_at_N = Dict{Int, Set{Tuple{Float64,Float64,Int}}}()
for N in N_VALUES
    settings_at_N[N] = Set((r.h, r.a, r.L) for r in rows_for_N(N))
end
common = intersect(values(settings_at_N)...)
common_sorted = sort(collect(common))

if isempty(common)
    println("  (no (h,a,L) appears at every N — skipping)")
else
    for (h, a, L) in common_sorted
        Ns   = Float64[]
        ts   = Float64[]
        for N in N_VALUES
            for r in rows_for_N(N)
                if r.h == h && r.a == a && r.L == L
                    push!(Ns, N); push!(ts, r.t_msm)
                end
            end
        end
        if length(Ns) ≥ 2
            slope = loglog_slope(Ns, ts)
            @printf "  (h=%.2f, a=%.2f, L=%d):  slope = %+5.2f   (N: %s, t_msm: %s)\n" h a L slope join((string(Int(x)) for x in Ns), ", ") join((@sprintf("%.4g",t) for t in ts), ", ")
        end
    end
end

# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

println("\n", repeat("-", 72))
println("Writing plots to ", PLOT_DIR)
println(repeat("-", 72))

# 1. Accuracy vs cost — one panel per N, all points + Pareto front overlaid.
panels = []
for N in N_VALUES
    rs   = rows_for_N(N)
    front = pareto_front(rs)
    front_set = Set((r.h, r.a, r.L) for r in front)

    xs_all = [r.t_msm  for r in rs]
    ys_all = [r.rel_err for r in rs]
    xs_pf  = [r.t_msm  for r in front]
    ys_pf  = [r.rel_err for r in front]

    p = scatter(xs_all, ys_all;
                label = "all",
                xscale = :log10, yscale = :log10,
                xlabel = "t_msm (s)", ylabel = "rel_err",
                title  = "N = $N",
                markersize = 4, markercolor = :gray, markerstrokewidth = 0)
    scatter!(p, xs_pf, ys_pf;
             label = "Pareto",
             markersize = 6, markercolor = :red, markerstrokewidth = 0)
    plot!(p, xs_pf, ys_pf; label = "", linecolor = :red, linealpha = 0.5)
    push!(panels, p)
end
P1 = plot(panels...; layout = (length(N_VALUES), 1), size = (700, 220 * length(N_VALUES)),
          left_margin = 5Plots.mm)
savefig(P1, joinpath(PLOT_DIR, "$(PLOT_PREFIX)_accuracy_vs_cost.png"))

# 2. Error vs h, one curve per a, for largest N. Reference slope p = 3.
P2 = plot(; xscale = :log10, yscale = :log10,
          xlabel = "h (Å)", ylabel = "rel_err",
          title  = "Error vs h (N = $N_top, largest L per (h,a))",
          legend = :bottomright)
a_values = sort(unique(r.a for r in rs_top_L))
for a in a_values
    sub = sort([r for r in rs_top_L if r.a == a]; by = r -> r.h)
    if length(sub) ≥ 2
        plot!(P2, [r.h for r in sub], [r.rel_err for r in sub];
              marker = :circle, label = @sprintf("a = %.1f Å", a))
    end
end
# Reference slope p = 3: through the median point of the first series, if available.
let
    h_ref = [0.5, 2.0]
    e_ref_anchor = 1e-3   # arbitrary anchor; we just want to show the slope
    e_ref = e_ref_anchor .* (h_ref ./ h_ref[1]).^3
    plot!(P2, h_ref, e_ref; linestyle = :dash, linecolor = :black,
          label = "slope p = 3")
end
savefig(P2, joinpath(PLOT_DIR, "$(PLOT_PREFIX)_error_vs_h.png"))

# 3. Cost vs N for the (h, a, L) settings common to all N. Reference slope 1.
P3 = plot(; xscale = :log10, yscale = :log10,
          xlabel = "N", ylabel = "t_msm (s)",
          title  = "Cost scaling t_msm vs N",
          legend = :bottomright)
if !isempty(common_sorted)
    for (h, a, L) in common_sorted
        Ns = Float64[]; ts = Float64[]
        for N in N_VALUES
            for r in rows_for_N(N)
                if r.h == h && r.a == a && r.L == L
                    push!(Ns, N); push!(ts, r.t_msm)
                end
            end
        end
        if length(Ns) ≥ 2
            plot!(P3, Ns, ts;
                  marker = :circle,
                  label = @sprintf("h=%.1f, a=%.1f, L=%d", h, a, L))
        end
    end
    # Reference slope 1 (anchored at first common point).
    Ns0 = Float64[]; ts0 = Float64[]
    h0, a0, L0 = first(common_sorted)
    for N in N_VALUES, r in rows_for_N(N)
        if r.h == h0 && r.a == a0 && r.L == L0
            push!(Ns0, N); push!(ts0, r.t_msm)
        end
    end
    if length(Ns0) ≥ 2
        N_ref = [minimum(Ns0), maximum(Ns0)]
        t_ref = ts0[1] .* (N_ref ./ Ns0[1]).^1
        plot!(P3, N_ref, t_ref; linestyle = :dash, linecolor = :black,
              label = "slope 1 (O(N))")
    end
end
savefig(P3, joinpath(PLOT_DIR, "$(PLOT_PREFIX)_cost_vs_N.png"))

println("  - ", joinpath(PLOT_DIR, "$(PLOT_PREFIX)_accuracy_vs_cost.png"))
println("  - ", joinpath(PLOT_DIR, "$(PLOT_PREFIX)_error_vs_h.png"))
println("  - ", joinpath(PLOT_DIR, "$(PLOT_PREFIX)_cost_vs_N.png"))
println("\nDone.")
