"""
    top_level!(e, q, grid, splitting) -> e

Top-level grid potential (paper eq. 10):

    e[m] = Σ_n k_L(r_m − r_n) q[n]

evaluated as a direct sum over all grid pairs (no cutoff — `k_L` has no
compact support). Periodic axes apply minimum-image displacement.

When the grid is fully periodic the top grid is normally reduced to a
single point along each periodic axis (see `top_grid_size`); the charge
at that point is then set to zero by [`apply_neutralising_background!`]
if the splitting requires it.

`e` is overwritten.
"""
function top_level!(e::AbstractArray{T,D},
                    q::AbstractArray{T,D},
                    grid::UniformGrid{D,T},
                    splitting) where {D,T<:AbstractFloat}
    @assert size(e) == size(q) == grid.size
    fill!(e, zero(T))
    @inbounds for m in CartesianIndices(e)
        r_m = grid.origin .+ SVector{D,T}(ntuple(α -> T(m[α] - 1) * grid.spacing[α], Val(D)))
        acc = zero(T)
        for n in CartesianIndices(q)
            r_n = grid.origin .+ SVector{D,T}(ntuple(α -> T(n[α] - 1) * grid.spacing[α], Val(D)))
            Δr  = _periodic_displacement(r_m - r_n, grid)
            acc += top_level(splitting, Δr) * q[n]
        end
        e[m] = acc
    end
    return e
end

"""
    apply_neutralising_background!(q, splitting, grid) -> q

For splittings flagged with `requires_neutralising_background == true`
and a fully periodic grid, zero out the (single-point) top-level grid
charge to implement the neutralising-background trick (paper §2.1).

For partial periodicity or splittings that don't require it, this is a
no-op. Returns `q` for chaining.
"""
function apply_neutralising_background!(q::AbstractArray{T},
                                        splitting,
                                        grid::UniformGrid{D,T}) where {D,T}
    requires_neutralising_background(splitting) || return q
    all(grid.periodic) || return q
    fill!(q, zero(T))
    return q
end

"""
    top_grid_size(g_finest, L) -> NTuple{D,Int}

The MSM top-level grid extent, derived from the finest grid by halving
each axis `L−1` times. Periodic axes are clipped to a minimum of 1;
open axes are simply halved.
"""
function top_grid_size(g_finest::UniformGrid{D,T}, L::Int) where {D,T}
    ntuple(Val(D)) do α
        n = g_finest.size[α]
        for _ in 1:(L - 1)
            n = max(div(n, 2), 1)
        end
        n
    end
end

@inline function _periodic_displacement(Δr::SVector{D,T}, grid::UniformGrid{D,T}) where {D,T}
    out = Δr
    @inbounds for α in 1:D
        if grid.periodic[α]
            Lα = grid.spacing[α] * grid.size[α]
            out = setindex(out, out[α] - Lα * round(out[α] / Lα), α)
        end
    end
    return out
end
