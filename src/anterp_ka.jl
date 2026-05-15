# KernelAbstractions-backed gather operators on particles:
#   - interpolate_ka!       (paper eq. 12)
#   - interpolate_grad_ka!  (∇ on the dual basis side, used by forces)
#
# The scatter operator `anterpolate_ka!` (paper eq. 7) lives at the end of
# this file but is filled in by PR-C; the CPU version is in anterp.jl.
#
# Per-workitem dispatch = one particle. No race on the output, no atomics.

using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, Backend

# --- helpers --------------------------------------------------------------

# Same tensor-product helpers the CPU path uses, restated here without the
# @generated machinery so KA's kernel rewriter sees plain Julia. The
# compiler should still unroll on a concrete S, D.
@inline function _tensor_basis_value_ka(ϕ_per_axis::NTuple{D,Any},
                                         off::NTuple{D,Int},
                                         base::Int) where {D}
    v = one(eltype(ϕ_per_axis[1]))
    @inbounds for α in 1:D
        v *= ϕ_per_axis[α][off[α] + base]
    end
    return v
end

# --- interpolate (gather) -------------------------------------------------

@kernel function _interpolate_ka_kernel!(potentials,
                                          @Const(positions),
                                          @Const(grid_values),
                                          grid,
                                          basis,
                                          ::Val{D},
                                          ::Val{S}) where {D, S}
    p = @index(Global)
    T = eltype(potentials)

    ξ  = particle_to_grid(positions[p], grid)
    m0 = ntuple(α -> floor(Int, ξ[α]), Val(D))
    ϕ_per_axis = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                  ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                    Val(2S)),
                        Val(D))

    acc = zero(T)
    off_lo = -(S - 1)
    off_hi = S
    @inbounds for off in Iterators.product(ntuple(_ -> off_lo:off_hi, Val(D))...)
        idx = ntuple(α -> m0[α] + off[α], Val(D))
        idx_wrapped, in_bounds = wrap_index(idx, grid)
        if in_bounds
            bv = _tensor_basis_value_ka(ϕ_per_axis, off, S)
            acc += bv * grid_values[idx_wrapped...]
        end
    end
    @inbounds potentials[p] = acc
end

"""
    interpolate_ka!(potentials, positions, grid_values, grid, basis, backend)
        -> potentials

KernelAbstractions counterpart of `interpolate!` (paper eq. 12). One
workitem per particle; gathers grid potentials via basis weights and
writes a single output per particle (no race).

`backend` is a `KernelAbstractions.Backend` (e.g. `KA.CPU()`,
`CUDABackend()`).
"""
function interpolate_ka!(potentials::AbstractVector{T},
                         positions::AbstractVector{SVector{D,T}},
                         grid_values::AbstractArray{T,D},
                         grid::UniformGrid{D,T},
                         basis,
                         backend::Backend) where {D, T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(potentials) == length(positions)
    fill!(potentials, zero(T))
    S = support_radius(basis)
    N = length(positions)
    kernel = _interpolate_ka_kernel!(backend)
    kernel(potentials, positions, grid_values, grid, basis, Val(D), Val(S);
           ndrange = N)
    _ka_synchronize(backend)
    return potentials
end

# --- interpolate_grad (gather, gradient w.r.t. particle position) ---------

@kernel function _interpolate_grad_ka_kernel!(grads,
                                               @Const(positions),
                                               @Const(grid_values),
                                               grid,
                                               basis,
                                               inv_h,
                                               ::Val{D},
                                               ::Val{S}) where {D, S}
    p = @index(Global)
    T = eltype(inv_h)   # SVector{D,T} → T

    ξ  = particle_to_grid(positions[p], grid)
    m0 = ntuple(α -> floor(Int, ξ[α]), Val(D))
    ϕ_per_axis  = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                   ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                      Val(2S)),
                          Val(D))
    ϕp_per_axis = ntuple(α -> ntuple(k -> eval_phi_prime(basis,
                                                         ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                      Val(2S)),
                          Val(D))

    acc = zero(SVector{D,T})
    off_lo = -(S - 1)
    off_hi = S
    @inbounds for off in Iterators.product(ntuple(_ -> off_lo:off_hi, Val(D))...)
        idx = ntuple(α -> m0[α] + off[α], Val(D))
        idx_wrapped, in_bounds = wrap_index(idx, grid)
        if in_bounds
            # Gradient tensor product, unrolled in-line: component α is
            # inv_h[α] · ϕp_per_axis[α][off[α]+S]  · ∏_{β≠α} ϕ_per_axis[β][off[β]+S]
            g = SVector{D,T}(ntuple(Val(D)) do α
                v = inv_h[α]
                for β in 1:D
                    v *= (β == α) ?
                         ϕp_per_axis[β][off[β] + S] :
                         ϕ_per_axis[β][off[β] + S]
                end
                v
            end)
            acc += g * grid_values[idx_wrapped...]
        end
    end
    @inbounds grads[p] = acc
end

"""
    interpolate_grad_ka!(grads, positions, grid_values, grid, basis, backend)
        -> grads

KernelAbstractions counterpart of `interpolate_grad!`. One workitem per
particle, gathers gradient contributions via `Φ'` weights.
"""
function interpolate_grad_ka!(grads::AbstractVector{SVector{D,T}},
                              positions::AbstractVector{SVector{D,T}},
                              grid_values::AbstractArray{T,D},
                              grid::UniformGrid{D,T},
                              basis,
                              backend::Backend) where {D, T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(grads) == length(positions)
    fill!(grads, zero(SVector{D,T}))
    S = support_radius(basis)
    N = length(positions)
    inv_h = SVector{D,T}(ntuple(α -> one(T) / grid.spacing[α], Val(D)))
    kernel = _interpolate_grad_ka_kernel!(backend)
    kernel(grads, positions, grid_values, grid, basis, inv_h, Val(D), Val(S);
           ndrange = N)
    _ka_synchronize(backend)
    return grads
end

# --- anterpolate (scatter) -----------------------------------------------
#
# Scatter operator: multiple particles can target the same grid cell, so
# accumulation goes through atomic adds. KA's `@atomic` lifts to native
# atomics on GPU backends and to `Atomix` on the CPU backend.

using Atomix: Atomix

@kernel function _anterpolate_ka_kernel!(grid_values,
                                          @Const(positions),
                                          @Const(charges),
                                          grid,
                                          basis,
                                          ::Val{D},
                                          ::Val{S}) where {D, S}
    p = @index(Global)
    T = eltype(grid_values)

    ξ  = particle_to_grid(positions[p], grid)
    m0 = ntuple(α -> floor(Int, ξ[α]), Val(D))
    q  = charges[p]
    ϕ_per_axis = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                  ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                    Val(2S)),
                        Val(D))

    off_lo = -(S - 1)
    off_hi = S
    @inbounds for off in Iterators.product(ntuple(_ -> off_lo:off_hi, Val(D))...)
        idx = ntuple(α -> m0[α] + off[α], Val(D))
        idx_wrapped, in_bounds = wrap_index(idx, grid)
        if in_bounds
            bv = _tensor_basis_value_ka(ϕ_per_axis, off, S)
            # Linear index for the atomic update — `Atomix.@atomic` takes a
            # single ref-like getindex expression, so we collapse the D-tuple
            # of axis indices into a linear `LinearIndices` slot.
            li = LinearIndices(grid_values)[idx_wrapped...]
            Atomix.@atomic grid_values[li] += bv * q
        end
    end
end

"""
    anterpolate_ka!(grid_values, positions, charges, grid, basis, backend)
        -> grid_values

KernelAbstractions counterpart of `anterpolate!` (paper eq. 7). One
workitem per particle scatters charge to the basis-supported grid cells
via atomic adds — required because multiple particles can write to the
same cell.

`grid_values` is cleared before scattering.
"""
function anterpolate_ka!(grid_values::AbstractArray{T,D},
                         positions::AbstractVector{SVector{D,T}},
                         charges::AbstractVector{T},
                         grid::UniformGrid{D,T},
                         basis,
                         backend::Backend) where {D, T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(positions) == length(charges)
    fill!(grid_values, zero(T))
    S = support_radius(basis)
    N = length(positions)
    kernel = _anterpolate_ka_kernel!(backend)
    kernel(grid_values, positions, charges, grid, basis, Val(D), Val(S);
           ndrange = N)
    _ka_synchronize(backend)
    return grid_values
end
