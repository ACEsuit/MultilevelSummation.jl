"""
    build_stencil(splitting, l, h_grid) -> (stencil, smax)

Precompute the level-`l` grid-cutoff stencil for the given `splitting`
at grid spacing `h_grid`. Returns a `(2·smax+1)`-shaped tensor of stencil
values together with the per-axis half-width `smax`.

The level-`l` "long-range" kernel `k_l(r)` (`1 ≤ l ≤ L-1`) has compact
support `|r| ≤ a_{l+1} = 2^l · a`. The stencil radius is `smax_α =
ceil(a_{l+1} / h_grid_α)`.
"""
function build_stencil(splitting, l::Int, h_grid::SVector{D,T}) where {D,T<:AbstractFloat}
    a_lp1 = T(2)^l * splitting.a
    smax  = ntuple(α -> ceil(Int, a_lp1 / h_grid[α]), Val(D))
    dims  = ntuple(α -> 2 * smax[α] + 1, Val(D))
    stencil = zeros(T, dims...)
    @inbounds for I in CartesianIndices(stencil)
        off = ntuple(α -> I[α] - 1 - smax[α], Val(D))
        Δr  = SVector{D,T}(ntuple(α -> T(off[α]) * h_grid[α], Val(D)))
        stencil[I] = long_range_level(splitting, l, Δr)
    end
    return stencil, smax
end

"""
    grid_cutoff!(e, q, grid, splitting, l) -> e

Compute the level-`l` "grid cutoff" potential (paper eq. 9):

    e[m] = Σ_n k_l(r_m − r_n) q[n]

where the sum is over destination grid points `n` within the compact
support of `k_l`. Periodic axes wrap; open axes drop out-of-bounds
contributions.

`e` is overwritten.
"""
function grid_cutoff!(e::AbstractArray{T,D},
                      q::AbstractArray{T,D},
                      grid::UniformGrid{D,T},
                      splitting,
                      l::Int) where {D,T<:AbstractFloat}
    @assert size(e) == size(q) == grid.size
    stencil, smax = build_stencil(splitting, l, grid.spacing)
    return _convolve!(e, q, stencil, smax, grid)
end

# Generic stencil convolution used by both grid_cutoff! and top_level!.
function _convolve!(e::AbstractArray{T,D},
                    q::AbstractArray{T,D},
                    stencil::AbstractArray{T,D},
                    smax::NTuple{D,Int},
                    grid::UniformGrid{D,T}) where {D,T<:AbstractFloat}
    fill!(e, zero(T))
    @inbounds for m in CartesianIndices(e)
        acc = zero(T)
        for I in CartesianIndices(stencil)
            off   = ntuple(α -> I[α] - 1 - smax[α], Val(D))
            n_raw = ntuple(α -> m[α] - 1 + off[α], Val(D))
            n_wrapped, in_bounds = wrap_index(n_raw, grid)
            in_bounds || continue
            acc += stencil[I] * q[n_wrapped...]
        end
        e[m] = acc
    end
    return e
end
