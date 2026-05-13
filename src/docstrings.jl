# Top-level docstrings for generic functions whose detailed semantics are
# spread across methods. Documenter.jl picks up the generic-function symbol
# and renders it in the API reference.

"""
    grad(K, r::SVector{D,T}) -> SVector{D,T}

Gradient of the pair kernel `K` evaluated at displacement vector `r`. Each
shipped kernel (`InversePower{N,T}`, `RationalDecay{N,T}`) implements its
own method; user-defined kernels must provide one if they need force
evaluation through the MSM pipeline.
"""
function grad end

"""
    short_range(splitting, r::SVector{D,T}) -> T

The short-range kernel component `K_0(r)` of a kernel splitting, supported
on `|r| ≤ a` (the splitting's cutoff).
"""
function short_range end

"""
    long_range_level(splitting, l::Integer, r::SVector{D,T}) -> T

The level-`l` "middle" kernel component `K_l(r)` of a kernel splitting,
defined for `l = 1, …, L-1`. Compactly supported on `|r| ≤ 2^l a`.
"""
function long_range_level end

"""
    top_level(splitting, r::SVector{D,T}) -> T

The top-level kernel component `K_L(r)` of a kernel splitting. Not
compactly supported in general (handled by direct summation on the top
grid).
"""
function top_level end

"""
    short_range_grad(splitting, r::SVector{D,T}) -> SVector{D,T}

The gradient of [`short_range`](@ref) with respect to `r`.
"""
function short_range_grad end

"""
    long_range_level_grad(splitting, l::Integer, r::SVector{D,T}) -> SVector{D,T}

The gradient of [`long_range_level`](@ref) with respect to `r`.
"""
function long_range_level_grad end

"""
    top_level_grad(splitting, r::SVector{D,T}) -> SVector{D,T}

The gradient of [`top_level`](@ref) with respect to `r`.
"""
function top_level_grad end

"""
    requires_neutralising_background(splitting) -> Bool

Whether the splitting requires zeroing the (single-point) top-level grid
charge for fully periodic systems. `true` for the Coulomb-matched
`HardyC2Cubic` splitting; `false` for splittings of fast-decaying
kernels.
"""
function requires_neutralising_background end

"""
    level_scale(splitting, l::Integer) -> T

The kernel-scale parameter `a_l = 2^{l-1} · a` at level `l` of the
splitting.
"""
function level_scale end

"""
    eval_phi(basis, ξ::T) -> T

The 1-D interpolation basis function `Φ(ξ)`.
"""
function eval_phi end

"""
    eval_phi_prime(basis, ξ::T) -> T

The derivative of the 1-D interpolation basis: `Φ'(ξ)`.
"""
function eval_phi_prime end

"""
    support_radius(basis) -> Int

Half-width of the basis support in grid spacings. Each particle scatters
to `(2·support_radius)^D` grid points; each grid point gathers from the
same number of source points during prolongation / restriction.
"""
function support_radius end
