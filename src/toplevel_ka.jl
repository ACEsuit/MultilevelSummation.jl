# KernelAbstractions-backed top-level direct sum (paper eq. 10).
#
# O(N_top²) direct sum over the top grid. For fully-periodic Coulomb the
# top grid is 1×1×1 — the launch is trivial. Each destination cell is
# computed independently.

using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, Backend

@kernel function _top_level_ka_kernel!(e,
                                        @Const(q),
                                        grid,
                                        splitting,
                                        ::Val{D}) where {D}
    m = @index(Global, Cartesian)
    T = eltype(e)
    r_m = grid.origin .+ SVector{D,T}(ntuple(α -> T(m[α] - 1) * grid.spacing[α], Val(D)))
    acc = zero(T)
    @inbounds for n in CartesianIndices(q)
        r_n = grid.origin .+ SVector{D,T}(ntuple(α -> T(n[α] - 1) * grid.spacing[α], Val(D)))
        Δr  = _periodic_displacement(r_m - r_n, grid)
        acc += top_level(splitting, Δr) * q[n]
    end
    @inbounds e[m] = acc
end

"""
    top_level_ka!(e, q, grid, splitting, backend) -> e

KernelAbstractions counterpart of `top_level!`. One workitem per
destination cell, inner loop is over all source cells. Minimum-image
displacement applied per periodic axis via `_periodic_displacement`.
"""
function top_level_ka!(e::AbstractArray{T,D},
                       q::AbstractArray{T,D},
                       grid::UniformGrid{D,T},
                       splitting,
                       backend::Backend) where {D, T<:AbstractFloat}
    @assert size(e) == size(q) == grid.size
    fill!(e, zero(T))
    kernel = _top_level_ka_kernel!(backend)
    kernel(e, q, grid, splitting, Val(D); ndrange = size(e))
    _ka_synchronize(backend)
    return e
end

"""
    apply_neutralising_background_ka!(q, splitting, grid, backend) -> q

Same semantics as `apply_neutralising_background!` (paper §2.1): for
splittings that need it and a fully-periodic grid, zero out the
top-level charge. Implemented as a generic `fill!`, which dispatches to
the backend's array `fill!`.
"""
function apply_neutralising_background_ka!(q::AbstractArray{T},
                                           splitting,
                                           grid::UniformGrid{D,T},
                                           ::Backend) where {D, T}
    requires_neutralising_background(splitting) || return q
    all(grid.periodic) || return q
    fill!(q, zero(T))
    return q
end
