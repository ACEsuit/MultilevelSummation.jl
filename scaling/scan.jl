# End-to-end scaling study for MultilevelSummation.jl.
#
# Measures wall-clock cost of `msm_energy_forces` on realistic NaCl and
# H₂O fixtures from O(10²) to O(10⁵) atoms, at the same Pareto-optimal
# `(h, a)` choices the benchmark suite uses. The number of levels `L`
# is grown with system size so the top grid is always 1×1×1, leaving
# accuracy invariant across scales (see scaling/README.md).
#
# Each `(system, backend, N)` cell is run with a single warm-up call
# followed by up to 5 timed samples, recorded to a CSV in
# `scaling/results/`. The size sweep is budget-gated: if a cell's
# median time exceeds `MSM_SCALING_BUDGET_S` (default 60 s), larger
# sizes are skipped *for that backend* and the script moves on.
#
# Backend discovery follows the same pattern as `test/gpu/runtests.jl`:
# whichever of `CUDA` / `AMDGPU` / `Metal` / `oneAPI` is installed in
# `scaling/Project.toml` is used for the GPU pass. Without any of them
# the script runs CPU-only.
#
# Usage:
#   julia -t auto --project=scaling scaling/scan.jl --label cpu-auto
#
# Optional flags:
#   --label <name>             tag for the CSV filename (default: hostname-threads)
#   --systems nacl,h2o         comma-separated subset of systems (default: both)
#   --budget <seconds>         per-cell wall-clock budget (default: 60)
#   --skip-gpu                 force CPU-only even if a GPU framework is present
#   --output <path>            override the default CSV path
#   --no-accuracy-check        skip the smallest-N CPU↔GPU sanity comparison

using Pkg
using Statistics: median, mean
using Printf
using Random: MersenneTwister
using StaticArrays

using MultilevelSummation
using MultilevelSummation.Tune: build_nacl, build_h2o

# ---------------------------------------------------------------------
# CLI parsing (no Pkg deps for argparse — keep deps slim)
# ---------------------------------------------------------------------

function parse_args(args)
    cfg = Dict{Symbol,Any}(
        :label              => "$(gethostname())-t$(Threads.nthreads())",
        :systems            => [:nacl, :h2o],
        :budget             => parse(Float64, get(ENV, "MSM_SCALING_BUDGET_S", "60")),
        :skip_gpu           => false,
        :output             => nothing,
        :do_accuracy_check  => true,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--label"
            cfg[:label] = args[i+1]; i += 2
        elseif a == "--systems"
            cfg[:systems] = Symbol.(split(args[i+1], ","))
            i += 2
        elseif a == "--budget"
            cfg[:budget] = parse(Float64, args[i+1]); i += 2
        elseif a == "--skip-gpu"
            cfg[:skip_gpu] = true; i += 1
        elseif a == "--output"
            cfg[:output] = args[i+1]; i += 2
        elseif a == "--no-accuracy-check"
            cfg[:do_accuracy_check] = false; i += 1
        else
            error("Unknown flag: $a")
        end
    end
    return cfg
end

const CFG = parse_args(ARGS)

# ---------------------------------------------------------------------
# GPU framework discovery (mirror of test/gpu/runtests.jl)
# ---------------------------------------------------------------------

const project_deps = Set(keys(Pkg.project().dependencies))

const has_cuda   = "CUDA"   in project_deps
const has_amdgpu = "AMDGPU" in project_deps
const has_metal  = "Metal"  in project_deps
const has_oneapi = "oneAPI" in project_deps

if CFG[:skip_gpu]
    const GPU_AVAILABLE  = false
    const MAKE_GPU_ARR   = identity
    const GPU_FRAMEWORK  = "none"
elseif has_cuda
    using CUDA
    const GPU_AVAILABLE  = CUDA.functional()
    const MAKE_GPU_ARR   = CuArray
    const GPU_FRAMEWORK  = "CUDA"
    # Our short-range kernel uses `Ref`-boxed accumulators inside a
    # closure passed to `for_each_neighbour`. On a CUDA backend each
    # workitem heap-allocates a small box (a few bytes); CUDA's
    # default per-kernel malloc heap is only 8 MB, which is exhausted
    # somewhere around N ≈ 30 000. Bumping to 2 GB covers the full
    # sweep up to 262 144 atoms with margin. (See the
    # `Ref(zero(T))` accumulators in
    # `src/shortrange_ka.jl::_short_range_ka_kernel!`.)
    if GPU_AVAILABLE
        CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 2 * 1024^3)
    end
elseif has_amdgpu
    using AMDGPU
    const GPU_AVAILABLE  = AMDGPU.functional()
    const MAKE_GPU_ARR   = ROCArray
    const GPU_FRAMEWORK  = "AMDGPU"
elseif has_metal
    using Metal
    const GPU_AVAILABLE  = Metal.functional()
    const MAKE_GPU_ARR   = MtlArray
    const GPU_FRAMEWORK  = "Metal"
elseif has_oneapi
    using oneAPI
    const GPU_AVAILABLE  = oneAPI.functional()
    const MAKE_GPU_ARR   = oneArray
    const GPU_FRAMEWORK  = "oneAPI"
else
    const GPU_AVAILABLE  = false
    const MAKE_GPU_ARR   = identity
    const GPU_FRAMEWORK  = "none"
end

@info "Scaling scan starting" label = CFG[:label] threads = Threads.nthreads() gpu = GPU_FRAMEWORK gpu_ready = GPU_AVAILABLE budget_s = CFG[:budget]

# ---------------------------------------------------------------------
# Fixtures and hyperparameters
# ---------------------------------------------------------------------

# Pareto-optimal (h, a) matching benchmark/benchmarks.jl. The number of
# levels L is computed from the box so that the top grid is 1×1×1
# (handles the periodic image via the neutralising background).

const NACL_H = 2.0
const NACL_A = 4.0
const H2O_H  = 2.0
const H2O_A  = 8.0

# Size sweeps in powers of 2. NaCl: box = 4·n_super → box/h ∈ {4,8,16,32,64}.
# H2O: box ∈ {8,16,32,64,128} → box/h ∈ {4,8,16,32,64}.
const NACL_NSUPER_SWEEP = [2, 4, 8, 16, 32]
const H2O_BOX_SWEEP     = [8.0, 16.0, 32.0, 64.0, 128.0]

# Smallest sizes only used for the accuracy spot check (legacy CPU vs GPU).
const RTOL_ACCURACY = 1.0e-5

# Time sampling — abort early if a single sample exceeds budget.
const WARMUP_TIMEOUT_S    = 120.0   # warmup can take longer than budget for huge cases
const SAMPLE_TIMEOUT_MULT = 1.5     # if first sample > budget × this, no further samples

build_fixture(::Val{:nacl}, n_super) = build_nacl(n_super; σ = 0.1,
                                                  rng = MersenneTwister(0xBEEF))
build_fixture(::Val{:h2o},  box)     = build_h2o(box;
                                                  rng = MersenneTwister(0xBEEF))

hyperparams(::Val{:nacl}) = (h = NACL_H, a = NACL_A)
hyperparams(::Val{:h2o})  = (h = H2O_H,  a = H2O_A)

function fit_levels(box::Real, h::Real)
    ratio = box / h
    L = round(Int, log2(ratio)) + 1
    actual = 2.0 ^ (L - 1)
    isapprox(actual, ratio; atol = 1e-9) ||
        error("box/h = $ratio is not a power of 2; cannot pick L exactly. " *
              "Adjust the size sweep so box/h ∈ {2, 4, 8, …}.")
    return L
end

# ---------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------

"""
Run a single timed call to `msm_energy_forces`. Returns elapsed seconds
and the energy (for sanity tracking).
"""
function timed_call(positions, charges, cell, periodic, calc)
    # `@elapsed` measures wall clock; reset GC stats first to avoid
    # bias from prior cells.
    GC.gc()
    t = @elapsed (U, _F) = msm_energy_forces(positions, charges, cell, periodic, calc)
    return t, U
end

"""
Run warmup + up to 5 timed samples for one `(system, backend, N)` cell.
Bails after the first sample if it already exceeds `SAMPLE_TIMEOUT_MULT
× budget` (no point sampling further at sizes that won't make the cut).
"""
function measure(positions, charges, cell, periodic, calc; budget::Float64)
    # Warmup (not timed, but capped via a soft check below)
    t_warm, U_warm = timed_call(positions, charges, cell, periodic, calc)
    if t_warm > WARMUP_TIMEOUT_S
        return (samples = 0, min = t_warm, median = t_warm, mean = t_warm,
                U = U_warm, note = "warmup exceeded $(WARMUP_TIMEOUT_S)s")
    end

    times = Float64[]
    energies = Float64[]
    push!(times, NaN)   # placeholder for first timed sample
    push!(energies, NaN)

    for k in 1:5
        t, U = timed_call(positions, charges, cell, periodic, calc)
        if k == 1
            times[1] = t; energies[1] = U
            if t > SAMPLE_TIMEOUT_MULT * budget
                # Don't waste more samples at obviously over-budget cells.
                break
            end
        else
            push!(times, t); push!(energies, U)
        end
    end

    return (samples = length(times),
            min = minimum(times), median = median(times),
            mean = mean(times),   U = energies[1],
            note = "")
end

# ---------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------

const CSV_HEADER = ["system", "backend", "label", "threads", "framework",
                    "N", "box", "h", "a", "L",
                    "time_min", "time_median", "time_mean", "samples",
                    "U", "U_per_atom", "note"]

function csv_path(cfg)
    cfg[:output] === nothing || return cfg[:output]
    date  = string(Dates_today_iso())
    dir   = joinpath(@__DIR__, "results")
    isdir(dir) || mkpath(dir)
    return joinpath(dir, "scaling-$(cfg[:label])-$(date).csv")
end

# Stdlib Dates isn't auto-loaded; use the value the user sees in `date`.
# Lightweight stub avoids the import here.
function Dates_today_iso()
    # YYYY-MM-DD using `Libc.strftime` would need Libc; instead use
    # the system `date` command via `read`.
    s = read(`date -I`, String)
    return strip(s)
end

function write_row(io, row)
    function fmt(x)
        x === missing && return ""
        x isa AbstractFloat && return @sprintf("%.6g", x)
        return string(x)
    end
    println(io, join((fmt(row[k]) for k in CSV_HEADER), ","))
end

# ---------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------

function run_cell(system::Symbol, size_param, backend_label::String;
                  framework::String, budget::Float64)
    positions, charges, cell, periodic = build_fixture(Val(system), size_param)
    box    = cell[1, 1]
    hp     = hyperparams(Val(system))
    L      = fit_levels(box, hp.h)
    calc   = MSMCalculator(HardyC2Cubic(hp.a, L), CubicC1(), hp.h)
    N      = length(positions)

    if backend_label == "gpu"
        positions = MAKE_GPU_ARR(positions)
        charges   = MAKE_GPU_ARR(charges)
    end

    @info "Cell" system N backend = backend_label
    m = measure(positions, charges, cell, periodic, calc; budget = budget)

    row = Dict(
        "system"      => string(system),
        "backend"     => backend_label,
        "label"       => CFG[:label],
        "threads"     => Threads.nthreads(),
        "framework"   => backend_label == "gpu" ? framework : "OhMyThreads",
        "N"           => N,
        "box"         => box,
        "h"           => hp.h,
        "a"           => hp.a,
        "L"           => L,
        "time_min"    => m.samples > 0 ? m.min    : missing,
        "time_median" => m.samples > 0 ? m.median : missing,
        "time_mean"   => m.samples > 0 ? m.mean   : missing,
        "samples"     => m.samples,
        "U"           => m.U,
        "U_per_atom"  => m.U / N,
        "note"        => m.note,
    )
    return row, m
end

# ---------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------

function main()
    rows = Vector{Dict{String,Any}}()

    for system in CFG[:systems]
        sweep = system == :nacl ? NACL_NSUPER_SWEEP :
                system == :h2o  ? H2O_BOX_SWEEP     :
                error("unknown system: $system")
        for backend_label in ("cpu", "gpu")
            (backend_label == "gpu" && !GPU_AVAILABLE) && continue
            @info "Backend sweep" system backend_label
            cpu_gpu_accuracy_pair = Float64[]
            for size_param in sweep
                row, m = run_cell(system, size_param, backend_label;
                                   framework = GPU_FRAMEWORK,
                                   budget    = CFG[:budget])
                push!(rows, row)
                if m.samples > 0 && CFG[:do_accuracy_check] && size_param == sweep[1]
                    push!(cpu_gpu_accuracy_pair, m.U)
                end
                if m.samples == 0 || m.median > CFG[:budget]
                    @info("Budget exceeded; skipping larger sizes",
                          system, backend_label,
                          median_s = m.median, size_param = size_param)
                    push!(rows, _budget_skip_row(system, sweep, size_param,
                                                  backend_label,
                                                  hyperparams(Val(system))))
                    break
                end
            end
            # Accuracy spot check at smallest N
            if CFG[:do_accuracy_check] && length(cpu_gpu_accuracy_pair) == 1 &&
               backend_label == "gpu" && GPU_AVAILABLE
                U_cpu = _smallest_cpu_energy(rows, system)
                U_gpu = cpu_gpu_accuracy_pair[1]
                rel = abs(U_cpu - U_gpu) / max(abs(U_cpu), 1e-12)
                if rel <= RTOL_ACCURACY
                    @info "Accuracy spot check OK" system rel_err = rel
                else
                    @warn "Accuracy spot check FAILED at smallest N" system U_cpu U_gpu rel_err = rel
                end
            end
        end
    end

    out = csv_path(CFG)
    open(out, "w") do io
        println(io, join(CSV_HEADER, ","))
        for row in rows
            write_row(io, row)
        end
    end
    @info "Wrote $(length(rows)) rows" path = out
    return rows
end

function _budget_skip_row(system, sweep, last_attempted, backend, hp)
    # Emit one note-row at the first size we did NOT attempt, so the
    # CSV is self-documenting about where the sweep stopped.
    return Dict(
        "system"      => string(system),
        "backend"     => backend,
        "label"       => CFG[:label],
        "threads"     => Threads.nthreads(),
        "framework"   => backend == "gpu" ? GPU_FRAMEWORK : "OhMyThreads",
        "N"           => missing,
        "box"         => missing,
        "h"           => hp.h,
        "a"           => hp.a,
        "L"           => missing,
        "time_min"    => missing,
        "time_median" => missing,
        "time_mean"   => missing,
        "samples"     => 0,
        "U"           => missing,
        "U_per_atom"  => missing,
        "note"        => "budget gate triggered after $(last_attempted)",
    )
end

function _smallest_cpu_energy(rows, system)
    for row in rows
        if row["system"] == string(system) && row["backend"] == "cpu" &&
           row["samples"] > 0
            return row["U"]
        end
    end
    return missing
end

main()
