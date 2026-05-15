"""
    MultilevelSummation.Reference

Naive reference implementations for the Coulomb pair sum, used as
absolute baselines in tests and tuning:

- [`ewald_energy`](@ref) / [`ewald_energy_forces`](@ref) — 3D Ewald
  reference for fully-periodic orthorhombic cells. See `ewald.jl`.
- [`naive_energy`](@ref) / [`naive_energy_forces`](@ref) — kernel-generic
  `O(N²)` direct summation with optional periodic-image truncation.
  See `naive.jl`.

    using MultilevelSummation.Reference: ewald_energy, ewald_energy_forces
    using MultilevelSummation.Reference: naive_energy, naive_energy_forces
"""
module Reference

using StaticArrays
using SpecialFunctions
using OhMyThreads: tmapreduce
using ChunkSplitters: chunks
using ..MultilevelSummation: grad   # for naive_energy_forces kernel gradient
using ..MultilevelSummation: _assert_orthorhombic, _image_ranges, _shift

export ewald_energy, ewald_energy_forces, ewald_reference
export naive_energy, naive_energy_forces

include("ewald.jl")
include("naive.jl")

end # module Reference
