# tuning/plotting.jl
#
# Plot helpers for the tuning scripts. Lives in the tuning environment
# only — `Plots.jl` is too heavy to pull into the core package. If we
# later want `Tune.plot_summary` reachable from outside the tuning env,
# this would move into a package extension (`ext/MultilevelSummationPlotsExt.jl`).
#
# `include`d from `tune_NaCl.jl` and `tune_H2O.jl`. Expects `Plots` to
# be available in the calling environment.

using Plots
using Printf
using MultilevelSummation.Tune: pareto_front, SweepResult

# Lightweight conversion: rows have everything pareto_front needs.
_row_to_sweepresult(r) = SweepResult{Float64}(r.h, r.a, r.L, r.n_grid, NaN, r.rel_err, r.t_msm)

"""
    plot_sweep(rows, plot_dir, prefix; subtitle = "")

Produce the three diagnostic plots used in the tuning analysis:

- `<prefix>_accuracy_vs_cost.png` — log-log `rel_err` vs `t_msm`, one
  panel per `N`, Pareto front highlighted.
- `<prefix>_error_vs_h.png` — `rel_err` vs `h` at fixed `a`, on the
  largest `N`, with reference slope `p = 3`.
- `<prefix>_cost_vs_N.png` — `t_msm` vs `N` for any `(h, a, L)`
  that appears at every `N`, with reference slope 1.

`subtitle` is appended to each plot title (e.g. `"σ = 0.1 Å"` or
`"ρ = 0.0334 /Å³"`).
"""
function plot_sweep(rows, plot_dir::AbstractString, prefix::AbstractString;
                    subtitle::AbstractString = "")
    isdir(plot_dir) || mkpath(plot_dir)
    Ns       = sort(unique(r.N for r in rows))
    N_top    = maximum(Ns)
    titlesuf = isempty(subtitle) ? "" : " — $subtitle"

    # 1. Accuracy vs cost, one panel per N.
    panels = []
    for N in Ns
        group = [r for r in rows if r.N == N]
        front = pareto_front([_row_to_sweepresult(r) for r in group])
        front_set = Set((r.h, r.a, r.L) for r in front)

        xs_all = [r.t_msm  for r in group]
        ys_all = [r.rel_err for r in group]
        xs_pf  = [r.t_msm  for r in front]
        ys_pf  = [r.rel_err for r in front]

        p = scatter(xs_all, ys_all;
                    label = "all",
                    xscale = :log10, yscale = :log10,
                    xlabel = "t_msm (s)", ylabel = "rel_err",
                    title  = "N = $N$titlesuf",
                    markersize = 4, markercolor = :gray, markerstrokewidth = 0)
        scatter!(p, xs_pf, ys_pf;
                 label = "Pareto",
                 markersize = 6, markercolor = :red, markerstrokewidth = 0)
        plot!(p, xs_pf, ys_pf; label = "", linecolor = :red, linealpha = 0.5)
        push!(panels, p)
    end
    P1 = plot(panels...; layout = (length(Ns), 1), size = (700, 220 * length(Ns)),
              left_margin = 5Plots.mm)
    savefig(P1, joinpath(plot_dir, "$(prefix)_accuracy_vs_cost.png"))

    # 2. Error vs h, one line per a, on largest N. Largest L per (h, a).
    rs_top = [r for r in rows if r.N == N_top]
    keyed = Dict{Tuple{Float64,Float64}, NamedTuple}()
    for r in rs_top
        k = (r.h, r.a)
        if !haskey(keyed, k) || r.L > keyed[k].L
            keyed[k] = r
        end
    end
    rs_topL = collect(values(keyed))

    P2 = plot(; xscale = :log10, yscale = :log10,
              xlabel = "h (Å)", ylabel = "rel_err",
              title  = "Error vs h (N = $N_top$titlesuf)",
              legend = :bottomright)
    a_values = sort(unique(r.a for r in rs_topL))
    for a in a_values
        sub = sort([r for r in rs_topL if r.a == a]; by = r -> r.h)
        length(sub) ≥ 2 || continue
        plot!(P2, [r.h for r in sub], [r.rel_err for r in sub];
              marker = :circle, label = @sprintf("a = %.1f Å", a))
    end
    # Reference slope p = 3.
    let h_ref = [0.5, 2.0], anchor = 1e-3
        e_ref = anchor .* (h_ref ./ h_ref[1]).^3
        plot!(P2, h_ref, e_ref; linestyle = :dash, linecolor = :black,
              label = "slope p = 3")
    end
    savefig(P2, joinpath(plot_dir, "$(prefix)_error_vs_h.png"))

    # 3. Cost vs N for any (h, a, L) shared across all N.
    settings_at_N = Dict(N => Set((r.h, r.a, r.L) for r in rows if r.N == N) for N in Ns)
    common = isempty(Ns) ? Set{Tuple{Float64,Float64,Int}}() : intersect(values(settings_at_N)...)
    P3 = plot(; xscale = :log10, yscale = :log10,
              xlabel = "N", ylabel = "t_msm (s)",
              title  = "Cost scaling t_msm vs N$titlesuf",
              legend = :bottomright)
    if !isempty(common)
        for (h, a, L) in sort(collect(common))
            Ns_pts = Float64[]; ts = Float64[]
            for N in Ns, r in rows
                r.N == N || continue
                r.h == h && r.a == a && r.L == L || continue
                push!(Ns_pts, N); push!(ts, r.t_msm)
            end
            length(Ns_pts) ≥ 2 || continue
            plot!(P3, Ns_pts, ts;
                  marker = :circle,
                  label = @sprintf("h=%.1f, a=%.1f, L=%d", h, a, L))
        end
        # Reference slope 1.
        h0, a0, L0 = first(sort(collect(common)))
        Ns0 = Float64[]; ts0 = Float64[]
        for N in Ns, r in rows
            r.N == N || continue
            r.h == h0 && r.a == a0 && r.L == L0 || continue
            push!(Ns0, N); push!(ts0, r.t_msm)
        end
        if length(Ns0) ≥ 2
            N_ref = [minimum(Ns0), maximum(Ns0)]
            t_ref = ts0[1] .* (N_ref ./ Ns0[1])
            plot!(P3, N_ref, t_ref; linestyle = :dash, linecolor = :black,
                  label = "slope 1 (O(N))")
        end
    end
    savefig(P3, joinpath(plot_dir, "$(prefix)_cost_vs_N.png"))

    return (joinpath(plot_dir, "$(prefix)_accuracy_vs_cost.png"),
            joinpath(plot_dir, "$(prefix)_error_vs_h.png"),
            joinpath(plot_dir, "$(prefix)_cost_vs_N.png"))
end
