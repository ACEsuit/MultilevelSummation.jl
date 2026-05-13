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
interface. 

**Highly experimental** 
— the API is not stable
- only the Coulomb (`1/r`) splitting is currently implemented
- performance is not yet tuned (many allocations)
- no ChainRules integration
- missing GPU port via `KernelAbstractions.jl`

See the [documentation](https://ACEsuit.github.io/MultilevelSummation.jl/dev/) 
for details, examples, and the implementation plan.
