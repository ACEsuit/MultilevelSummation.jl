module MultilevelSummation

using StaticArrays
using OhMyThreads: tforeach, tmapreduce
using ChunkSplitters: chunks

# Top-level docstrings for generic function symbols
include("docstrings.jl")

# Phase 1: math primitives ----------------------------------------------

# kernels
include("kernels/inverse_power.jl")
include("kernels/rational_decay.jl")

# interpolation basis
include("basis/cubic.jl")

# splittings (Coulomb-only for now)
include("splittings/hardy_c2cubic.jl")

# Orthorhombic-cell + periodic-image helpers shared between core.jl
# and the Reference submodule.
include("cell_helpers.jl")

# Phase 3: grid + anterpolation / interpolation
include("grid.jl")
include("anterp.jl")

# Phase 4: restriction / prolongation
include("transfer.jl")

# Phase 5: grid-cutoff convolution
include("gridcutoff.jl")

# Phase 6: top-level direct sum
include("toplevel.jl")

# Phase 7: end-to-end MSM assembly + calculator
include("core.jl")

# Phase 8: AtomsCalculators / AtomsBase wrapper
include("calculator.jl")

export InversePower, Coulomb, RationalDecay
export CubicC1, eval_phi, eval_phi_prime, support_radius
export HardyC2Cubic, short_range, long_range_level, top_level
export short_range_grad, long_range_level_grad, top_level_grad
export requires_neutralising_background, level_scale
export UniformGrid, grid_zeros, npoints, wrap_index, particle_to_grid
export anterpolate!, interpolate!, interpolate_grad!
export restrict!, prolong!, coarser_grid
export grid_cutoff!, build_stencil
export top_level!, apply_neutralising_background!, top_grid_size
export MSMCalculator, msm_energy, msm_energy_forces
export build_grid_hierarchy, kernel_self_value
export grad

# --------------------------------------------------------------------------
# Public submodules — must come last so they can `using ..MultilevelSummation`
# --------------------------------------------------------------------------

# Reference implementations (Ewald, naive direct sum) used by tests and tuning.
include("reference/Reference.jl")

# Programmatic hyperparameter sweeps.
include("tune/Tune.jl")

end # module MultilevelSummation
