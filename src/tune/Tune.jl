"""
    MultilevelSummation.Tune

Hyperparameter exploration plus a small set of realistic test
configurations used by the tuning scripts, the benchmark suite, and the
test suite as common fixtures.

API surface:

- **System builders** — [`build_nacl`](@ref), [`build_h2o`](@ref).
  *Experimental staging area* for an upstream PR to
  [AtomsBuilder.jl](https://github.com/JuliaMolSim/AtomsBuilder.jl);
  they may move or change shape. See `systems.jl`.
- **Sweep orchestration** — [`sweep`](@ref), [`run_system_sweep`](@ref),
  Pareto-front + recommend helpers, CSV writer. See `sweep.jl`.
- **Reference energy** — [`ewald_reference`](@ref) (re-exported from
  `MultilevelSummation.Reference`) — an auto-tuned convenience wrapper
  around `Reference.ewald_energy`. Used by `run_system_sweep`.
- **Text summaries** — [`print_pareto_per_N`](@ref),
  [`print_recommendations`](@ref), [`print_scaling`](@ref),
  [`print_summary`](@ref). Pure stdlib, no plotting deps. See
  `summary.jl`.
"""
module Tune

using Printf
using Random
using LinearAlgebra: norm
using StaticArrays
using ..MultilevelSummation
using ..MultilevelSummation: HardyC2Cubic, CubicC1, MSMCalculator, msm_energy
using ..MultilevelSummation.Reference: ewald_energy, ewald_reference

export SweepResult, sweep, pareto_front, recommend, ewald_reference
export build_nacl, build_h2o
export run_system_sweep, write_csv
export print_pareto_per_N, print_recommendations, print_scaling, print_summary

include("sweep.jl")             # SweepResult, sweep, pareto_front, recommend,
                                # run_system_sweep, write_csv
include("systems.jl")           # build_nacl, build_h2o + Poisson-disk helpers
include("summary.jl")           # print_* text reporting helpers
# `ewald_reference` lives in `src/reference/ewald.jl` and is re-exported above.

end # module Tune
