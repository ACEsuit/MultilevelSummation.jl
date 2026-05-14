"""
    restrict!(dst, src, dst_grid, src_grid, basis) -> dst

Grid-to-grid restriction (paper eq. 8). For each destination grid
point, gather contributions from the surrounding source grid points,
weighted by `Φ((r_src − r_dst)/h_dst)`.

The canonical MSM restriction is the special case where `dst_grid` has
exactly twice the spacing of `src_grid` (a "coarsen by 2" step); the
implementation is generic in the two grids' spacings and origins.

This is the gather form of the operator — each output cell `dst[n_dst]`
is computed independently from `src`. Threaded over destination points
via OhMyThreads. Periodic axes wrap via `wrap_index`; open axes drop
out-of-range contributions.

`dst` is overwritten.
"""
function restrict!(dst::AbstractArray{T,D},
                   src::AbstractArray{T,D},
                   dst_grid::UniformGrid{D,T},
                   src_grid::UniformGrid{D,T},
                   basis) where {D,T<:AbstractFloat}
    @assert size(dst) == dst_grid.size
    @assert size(src) == src_grid.size
    s = support_radius(basis)
    # In src-grid units the basis support around r_dst spans
    # `s · h_dst / h_src` per axis (for the canonical h_dst = 2·h_src
    # coarsen-by-2 this is 2s).
    half_w = ntuple(α -> ceil(Int, s * dst_grid.spacing[α] / src_grid.spacing[α]), Val(D))

    tforeach(CartesianIndices(dst)) do n_dst
        r_dst = dst_grid.origin .+ SVector{D,T}(ntuple(α -> T(n_dst[α] - 1) * dst_grid.spacing[α], Val(D)))
        # Window centre in src-grid coords (0-indexed; `wrap_index` expects
        # 0-based input). The gather range covers `m0 + off` for `off ∈
        # -half_w[α]:half_w[α]`.
        m0 = SVector{D,Int}(ntuple(α -> round(Int, (r_dst[α] - src_grid.origin[α]) / src_grid.spacing[α]), Val(D)))
        acc = zero(T)
        @inbounds for off in Iterators.product(ntuple(α -> -half_w[α]:half_w[α], Val(D))...)
            n_src = ntuple(α -> m0[α] + off[α], Val(D))
            n_src_wrapped, in_bounds = wrap_index(n_src, src_grid)
            in_bounds || continue
            bv = one(T)
            for α in 1:D
                # n_src is the 0-indexed src position (matching wrap_index's
                # convention); the corresponding particle-coord position is
                # `origin + n_src · spacing`.
                r_src_α = src_grid.origin[α] + T(n_src[α]) * src_grid.spacing[α]
                bv *= eval_phi(basis, (r_src_α - r_dst[α]) / dst_grid.spacing[α])
            end
            acc += bv * src[n_src_wrapped...]
        end
        @inbounds dst[n_dst] = acc
    end
    return dst
end

"""
    prolong!(dst, src, dst_grid, src_grid, basis) -> dst

Grid-to-grid prolongation (paper eq. 11). For each destination grid
point, gather from source grid points within the basis support window,
weighted by `Φ((r_dst − x_src)/h_src)`.

The transpose of `restrict!`. Canonical MSM prolongation is the special
case where `src_grid` has exactly twice the spacing of `dst_grid`.

`dst` is overwritten (not accumulated). Callers wanting to add to an
existing field should do `dst .+= prolong(...)` themselves.
"""
function prolong!(dst::AbstractArray{T,D},
                  src::AbstractArray{T,D},
                  dst_grid::UniformGrid{D,T},
                  src_grid::UniformGrid{D,T},
                  basis) where {D,T<:AbstractFloat}
    @assert size(dst) == dst_grid.size
    @assert size(src) == src_grid.size
    s = support_radius(basis)
    off_range = ntuple(_ -> -(s - 1):s, Val(D))

    tforeach(CartesianIndices(dst)) do n_dst
        r_dst = dst_grid.origin .+ SVector{D,T}(ntuple(α -> T(n_dst[α] - 1) * dst_grid.spacing[α], Val(D)))
        ξ = (r_dst - src_grid.origin) ./ src_grid.spacing
        m0 = SVector{D,Int}(ntuple(α -> floor(Int, ξ[α]), Val(D)))
        acc = zero(T)
        @inbounds for off in Iterators.product(off_range...)
            idx = ntuple(α -> m0[α] + off[α], Val(D))
            idx_wrapped, in_bounds = wrap_index(idx, src_grid)
            in_bounds || continue
            bv = one(T)
            for α in 1:D
                bv *= eval_phi(basis, ξ[α] - T(idx[α]))
            end
            acc += bv * src[idx_wrapped...]
        end
        @inbounds dst[n_dst] = acc
    end
    return dst
end

"""
    coarser_grid(g, factor=2) -> UniformGrid

Build the "next-coarser" grid in the MSM hierarchy: same origin, same
periodicity, each axis spacing scaled by `factor`, each axis extent
divided by `factor`. For periodic axes the original extent must be
divisible by `factor`. Open axes round down.
"""
function coarser_grid(g::UniformGrid{D,T,Per,Sz},
                      factor::Int = 2) where {D,T,Per,Sz}
    new_spacing = g.spacing .* T(factor)
    new_size = ntuple(Val(D)) do α
        if Per[α]
            rem(Sz[α], factor) == 0 ||
                throw(ArgumentError("axis $α extent $(Sz[α]) not divisible by $factor"))
            div(Sz[α], factor)
        else
            div(Sz[α], factor)
        end
    end
    return UniformGrid{D,T,Per,new_size}(new_spacing, g.origin)
end
