# KernelAbstractions-backed grid-cutoff convolution (paper eq. 9).
#
# The CPU `_convolve!` uses a `@generated` inner function that splices
# `(Per, Sz)` into per-axis unrolled wrap/clip code. Here we rely on
# `wrap_index` — itself `@generated` on the grid's type parameters — for
# the same effect; the kernel body remains plain Julia. Each destination
# cell is computed independently, no atomics needed.

using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, Backend
using Adapt: Adapt

@kernel function _convolve_ka_kernel!(e,
                                       @Const(q),
                                       @Const(stencil),
                                       smax,
                                       grid,
                                       ::Val{D}) where {D}
    m = @index(Global, Cartesian)
    T = eltype(e)
    acc = zero(T)
    @inbounds for I in CartesianIndices(stencil)
        idx_raw = ntuple(α -> (m[α] - 1) + (I[α] - 1 - smax[α]), Val(D))
        n_wrapped, in_bounds = wrap_index(idx_raw, grid)
        if in_bounds
            acc += stencil[I] * q[n_wrapped...]
        end
    end
    @inbounds e[m] = acc
end

"""
    grid_cutoff_ka!(e, q, grid, splitting, l, backend) -> e

KernelAbstractions counterpart of `grid_cutoff!`. Builds the stencil on
the host (it is small — one-shot per level), `Adapt`s it onto `backend`,
then launches one workitem per destination cell.
"""
function grid_cutoff_ka!(e::AbstractArray{T,D},
                         q::AbstractArray{T,D},
                         grid::UniformGrid{D,T},
                         splitting,
                         l::Int,
                         backend::Backend) where {D, T<:AbstractFloat}
    @assert size(e) == size(q) == grid.size
    stencil_host, smax = build_stencil(splitting, l, grid.spacing)
    stencil = Adapt.adapt(backend, stencil_host)
    fill!(e, zero(T))
    kernel = _convolve_ka_kernel!(backend)
    kernel(e, q, stencil, smax, grid, Val(D); ndrange = size(e))
    KernelAbstractions.synchronize(backend)
    return e
end
