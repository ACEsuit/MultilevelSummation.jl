# KernelAbstractions-backed restriction and prolongation.
#
# Both operators are already in gather form in `transfer.jl`: each
# destination cell is computed independently, no race. The KA path is a
# direct translation, one workitem per destination cell.

using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, Backend

# --- restrict -------------------------------------------------------------

@kernel function _restrict_ka_kernel!(dst,
                                       @Const(src),
                                       dst_grid,
                                       src_grid,
                                       basis,
                                       half_w,
                                       ::Val{D}) where {D}
    n_dst = @index(Global, Cartesian)
    T = eltype(dst)

    r_dst = dst_grid.origin .+ SVector{D,T}(ntuple(α -> T(n_dst[α] - 1) * dst_grid.spacing[α], Val(D)))
    m0    = ntuple(α -> round(Int, (r_dst[α] - src_grid.origin[α]) / src_grid.spacing[α]), Val(D))

    acc = zero(T)
    @inbounds for off in Iterators.product(ntuple(α -> -half_w[α]:half_w[α], Val(D))...)
        n_src = ntuple(α -> m0[α] + off[α], Val(D))
        n_src_wrapped, in_bounds = wrap_index(n_src, src_grid)
        if in_bounds
            bv = one(T)
            for α in 1:D
                r_src_α = src_grid.origin[α] + T(n_src[α]) * src_grid.spacing[α]
                bv *= eval_phi(basis, (r_src_α - r_dst[α]) / dst_grid.spacing[α])
            end
            acc += bv * src[n_src_wrapped...]
        end
    end
    @inbounds dst[n_dst] = acc
end

"""
    restrict_ka!(dst, src, dst_grid, src_grid, basis, backend) -> dst

KernelAbstractions counterpart of `restrict!` (paper eq. 8). Each
destination grid cell is computed independently — no atomics needed.
"""
function restrict_ka!(dst::AbstractArray{T,D},
                      src::AbstractArray{T,D},
                      dst_grid::UniformGrid{D,T},
                      src_grid::UniformGrid{D,T},
                      basis,
                      backend::Backend) where {D, T<:AbstractFloat}
    @assert size(dst) == dst_grid.size
    @assert size(src) == src_grid.size
    s = support_radius(basis)
    half_w = ntuple(α -> ceil(Int, s * dst_grid.spacing[α] / src_grid.spacing[α]), Val(D))
    kernel = _restrict_ka_kernel!(backend)
    kernel(dst, src, dst_grid, src_grid, basis, half_w, Val(D);
           ndrange = size(dst))
    _ka_synchronize(backend)
    return dst
end

# --- prolong --------------------------------------------------------------

@kernel function _prolong_ka_kernel!(dst,
                                      @Const(src),
                                      dst_grid,
                                      src_grid,
                                      basis,
                                      ::Val{D},
                                      ::Val{S}) where {D, S}
    n_dst = @index(Global, Cartesian)
    T = eltype(dst)

    r_dst = dst_grid.origin .+ SVector{D,T}(ntuple(α -> T(n_dst[α] - 1) * dst_grid.spacing[α], Val(D)))
    ξ     = (r_dst - src_grid.origin) ./ src_grid.spacing
    m0    = ntuple(α -> floor(Int, ξ[α]), Val(D))

    acc = zero(T)
    off_lo = -(S - 1)
    off_hi = S
    @inbounds for off in Iterators.product(ntuple(_ -> off_lo:off_hi, Val(D))...)
        idx = ntuple(α -> m0[α] + off[α], Val(D))
        idx_wrapped, in_bounds = wrap_index(idx, src_grid)
        if in_bounds
            bv = one(T)
            for α in 1:D
                bv *= eval_phi(basis, ξ[α] - T(idx[α]))
            end
            acc += bv * src[idx_wrapped...]
        end
    end
    @inbounds dst[n_dst] = acc
end

"""
    prolong_ka!(dst, src, dst_grid, src_grid, basis, backend) -> dst

KernelAbstractions counterpart of `prolong!` (paper eq. 11). One workitem
per destination cell, gather from source.
"""
function prolong_ka!(dst::AbstractArray{T,D},
                     src::AbstractArray{T,D},
                     dst_grid::UniformGrid{D,T},
                     src_grid::UniformGrid{D,T},
                     basis,
                     backend::Backend) where {D, T<:AbstractFloat}
    @assert size(dst) == dst_grid.size
    @assert size(src) == src_grid.size
    S = support_radius(basis)
    kernel = _prolong_ka_kernel!(backend)
    kernel(dst, src, dst_grid, src_grid, basis, Val(D), Val(S);
           ndrange = size(dst))
    _ka_synchronize(backend)
    return dst
end
