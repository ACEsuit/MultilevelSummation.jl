# tuning/tune_H2O.jl
#
# Hyperparameter sweep on a TIP3P-like liquid-water box at standard
# density (≈ 1 g/cm³). Oxygens placed by Poisson-disk rejection
# sampling, molecules given uniform random SO(3) orientations — see
# `MultilevelSummation.Tune.build_h2o` for details.
#
# Thin driver: configuration + `Tune.run_system_sweep` + summary +
# CSV + plots. All scaffolding lives in `MultilevelSummation.Tune` and
# the local `plotting.jl`.

using MultilevelSummation
using MultilevelSummation.Tune
using Random

include(joinpath(@__DIR__, "plotting.jl"))

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
const BOX_LENGTHS = (16.0, 20.0, 24.0)              # Å, gives ~411/801/1386 charges
const DENSITY     = 0.0334                          # molecules/Å³ ≈ 1 g/cm³
const D_MIN_OO    = 2.7                             # Å, oxygen exclusion radius
const H_VALUES    = (0.5, 1.0, 2.0)
const A_VALUES    = (2.0, 4.0, 8.0)
const RNG_SEED    = 42

# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
builder(box) = build_h2o(box; ρ = DENSITY, d_min = D_MIN_OO,
                         rng = MersenneTwister(RNG_SEED))

rows = run_system_sweep(builder, BOX_LENGTHS;
                        h_values   = H_VALUES,
                        a_values   = A_VALUES,
                        size_label = "box")

# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
print_summary(rows)

csv_path = joinpath(@__DIR__, "tune_H2O_results.csv")
write_csv(rows, csv_path)

plot_dir = joinpath(@__DIR__, "plots")
subtitle = "ρ = $DENSITY /Å³"
plots = plot_sweep(rows, plot_dir, "h2o"; subtitle = subtitle)

@info "wrote $(length(rows)) rows to $csv_path"
for p in plots
    @info "wrote plot $p"
end
