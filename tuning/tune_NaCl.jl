# tuning/tune_NaCl.jl
#
# Hyperparameter sweep on rock-salt NaCl supercells. Set SIGMA = 0.0
# to recover the perfect lattice; SIGMA ≈ 0.1 Å roughly mimics a 300 K
# thermal sample. The N-invariance of relative error you see at
# SIGMA = 0 is a perfect-lattice artefact — see the perturbed runs in
# `tune_NaCl_results.csv` (or any SIGMA > 0 run) for the realistic case.
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
const SIGMA        = 0.1                            # Å; 0.0 ⇒ perfect lattice
const N_SUPER_LIST = (3, 4, 5, 6, 8)                # n_super = 3..12 spans N ∈ [216, 13824]
const H_VALUES     = (0.5, 1.0, 2.0)
const A_VALUES     = (2.0, 4.0, 8.0)
const RNG_SEED     = 42

# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
builder(n) = build_nacl(n; σ = SIGMA, rng = MersenneTwister(RNG_SEED))

rows = run_system_sweep(builder, N_SUPER_LIST;
                        h_values   = H_VALUES,
                        a_values   = A_VALUES,
                        size_label = "n_super")

# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
print_summary(rows)

csv_path = joinpath(@__DIR__, "tune_NaCl_results.csv")
write_csv(rows, csv_path)

plot_dir = joinpath(@__DIR__, "plots")
subtitle = "σ = $SIGMA Å"
plots = plot_sweep(rows, plot_dir, "nacl"; subtitle = subtitle)

@info "wrote $(length(rows)) rows to $csv_path"
for p in plots
    @info "wrote plot $p"
end
