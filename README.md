# MultilevelSummation.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/MultilevelSummation.jl/dev/)
[![CI](https://github.com/ACEsuit/MultilevelSummation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ACEsuit/MultilevelSummation.jl/actions/workflows/CI.yml)

A Julia implementation of the **Multilevel Summation Method (MSM)** of
Hardy et al. (*J. Chem. Theory Comput.* **11**, 766–779, 2015) for fast
evaluation of long-range pair interactions, with support for any
dimension `d ∈ {1, 2, 3}`, per-axis `:open` / `:periodic` boundary
conditions, pluggable kernel splittings and interpolation bases, and an
[AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) +
[AtomsCalculators](https://github.com/JuliaMolSim/AtomsCalculators.jl)
interface. Ships with a naive 3D Ewald reference
(`MultilevelSummation.Reference`) and a programmatic hyperparameter
sweep API (`MultilevelSummation.Tune`) for accuracy-vs-cost analysis.

**Highly experimental** 
— the API is not stable
- only the Coulomb (`1/r`) splitting is currently implemented
- missing GPU port via `KernelAbstractions.jl`
- provides forces, but no ChainRules integration yet

The hot operators are multi-threaded via
[OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl). For
best performance launch Julia with `julia -t auto` (or set
`JULIA_NUM_THREADS`).

See the [documentation](https://ACEsuit.github.io/MultilevelSummation.jl/dev/) 
for details, examples, and the implementation plan.
