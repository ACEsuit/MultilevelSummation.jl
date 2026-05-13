# --- per-axis-unrolled helpers ---------------------------------------------
#
# Two small `@generated` functions replace `for α in 1:D` loops over axes.
# They take the per-axis basis-value tuples and a stencil offset, and emit
# a fully unrolled product (or gradient) at compile time.

"""
Unrolled tensor product of per-axis basis values:

    ∏_{α=1..D}  ϕ_per_axis[α][ off[α] + base ]
"""
@generated function _tensor_basis_value(ϕ_per_axis::NTuple{D,Any},
                                         off::NTuple{D,Int},
                                         base::Int) where {D}
    factors = [:(@inbounds ϕ_per_axis[$α][off[$α] + base]) for α in 1:D]
    return Expr(:call, :*, factors...)
end

"""
Unrolled tensor-product gradient. Component `α` of the returned SVector is

    inv_h[α] · ϕp_per_axis[α][off[α]+base]  ·  ∏_{β≠α}  ϕ_per_axis[β][off[β]+base]
"""
@generated function _tensor_basis_gradient(ϕ_per_axis::NTuple{D,Any},
                                            ϕp_per_axis::NTuple{D,Any},
                                            off::NTuple{D,Int},
                                            base::Int,
                                            inv_h::SVector{D,T}) where {D,T}
    comps = Expr[]
    for α in 1:D
        terms = Expr[]
        for β in 1:D
            t = (β == α) ?
                :(@inbounds ϕp_per_axis[$β][off[$β] + base]) :
                :(@inbounds ϕ_per_axis[$β][off[$β] + base])
            push!(terms, t)
        end
        push!(comps, :($(Expr(:call, :*, terms...)) * inv_h[$α]))
    end
    return :(SVector{$D,$T}($(comps...)))
end

# --- public operators ------------------------------------------------------
#
# Each public operator is a small wrapper that pulls `s = support_radius(basis)`
# and forwards to an `_impl!` function with `s` lifted into a `Val{S}` type
# parameter. The `_impl!` body is therefore fully type-stable on `S`, and
# `Val(2S)` / `Val(D)` allocations of the per-axis basis tuples produce
# concretely-typed `NTuple{2S, T}` data. The `@generated` helpers above then
# operate on a concrete type.
#
# For `basis::CubicC1`, `support_radius` is a small constant method that the
# compiler const-propagates, so the `Val(s)` step in the outer function is
# free at the call site.

"""
    anterpolate!(grid_values, positions, charges, grid, basis) -> grid_values

Anterpolation (paper eq. 7): scatter particle charges to the grid using
basis function `Φ` evaluated at offsets `(r_i − x_m) / h`. For each
particle, contributions are added to all grid points within the basis
support window.

Periodic axes wrap; open axes silently drop out-of-range contributions.

`grid_values` is *cleared* before scattering. Operates in-place; returns
`grid_values`.
"""
@inline function anterpolate!(grid_values::AbstractArray{T,D},
                              positions::AbstractVector{SVector{D,T}},
                              charges::AbstractVector{T},
                              grid::UniformGrid{D,T},
                              basis) where {D,T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(positions) == length(charges)
    fill!(grid_values, zero(T))
    return _anterpolate_impl!(grid_values, positions, charges, grid, basis,
                              Val(support_radius(basis)))
end

function _anterpolate_impl!(grid_values::AbstractArray{T,D},
                            positions::AbstractVector{SVector{D,T}},
                            charges::AbstractVector{T},
                            grid::UniformGrid{D,T},
                            basis,
                            ::Val{S}) where {D, T<:AbstractFloat, S}
    off_range = ntuple(_ -> -(S - 1):S, Val(D))

    @inbounds for p in eachindex(positions)
        ξ = particle_to_grid(positions[p], grid)
        q = charges[p]
        m0 = SVector{D,Int}(ntuple(α -> floor(Int, ξ[α]), Val(D)))
        # Per-axis basis values for the window — concretely `NTuple{2S, T}`.
        ϕ_per_axis = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                      ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                        Val(2S)),
                            Val(D))
        for off in Iterators.product(off_range...)
            idx = ntuple(α -> m0[α] + off[α], Val(D))
            idx_wrapped, in_bounds = wrap_index(idx, grid)
            in_bounds || continue
            bv = _tensor_basis_value(ϕ_per_axis, off, S)
            grid_values[idx_wrapped...] += bv * q
        end
    end
    return grid_values
end

"""
    interpolate!(potentials, positions, grid_values, grid, basis) -> potentials

Interpolation (paper eq. 12): gather grid potentials onto particle sites
using basis function `Φ`. The transpose of `anterpolate!`.

`potentials` is overwritten with the gathered values. Returns
`potentials`.
"""
@inline function interpolate!(potentials::AbstractVector{T},
                              positions::AbstractVector{SVector{D,T}},
                              grid_values::AbstractArray{T,D},
                              grid::UniformGrid{D,T},
                              basis) where {D,T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(potentials) == length(positions)
    fill!(potentials, zero(T))
    return _interpolate_impl!(potentials, positions, grid_values, grid, basis,
                              Val(support_radius(basis)))
end

function _interpolate_impl!(potentials::AbstractVector{T},
                            positions::AbstractVector{SVector{D,T}},
                            grid_values::AbstractArray{T,D},
                            grid::UniformGrid{D,T},
                            basis,
                            ::Val{S}) where {D, T<:AbstractFloat, S}
    off_range = ntuple(_ -> -(S - 1):S, Val(D))

    @inbounds for p in eachindex(positions)
        ξ = particle_to_grid(positions[p], grid)
        m0 = SVector{D,Int}(ntuple(α -> floor(Int, ξ[α]), Val(D)))
        ϕ_per_axis = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                      ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                        Val(2S)),
                            Val(D))
        acc = zero(T)
        for off in Iterators.product(off_range...)
            idx = ntuple(α -> m0[α] + off[α], Val(D))
            idx_wrapped, in_bounds = wrap_index(idx, grid)
            in_bounds || continue
            bv = _tensor_basis_value(ϕ_per_axis, off, S)
            acc += bv * grid_values[idx_wrapped...]
        end
        potentials[p] = acc
    end
    return potentials
end

"""
    interpolate_grad!(grads, positions, grid_values, grid, basis) -> grads

Gradient of `interpolate` with respect to particle position: returns the
field gradient at each particle site, `∇e_i = Σ_m ∇Φ((r_i−x_m)/h)·e_m / h`.

Used by force evaluation (∇ on the dual basis side).
"""
@inline function interpolate_grad!(grads::AbstractVector{SVector{D,T}},
                                   positions::AbstractVector{SVector{D,T}},
                                   grid_values::AbstractArray{T,D},
                                   grid::UniformGrid{D,T},
                                   basis) where {D,T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(grads) == length(positions)
    fill!(grads, zero(SVector{D,T}))
    return _interpolate_grad_impl!(grads, positions, grid_values, grid, basis,
                                   Val(support_radius(basis)))
end

function _interpolate_grad_impl!(grads::AbstractVector{SVector{D,T}},
                                 positions::AbstractVector{SVector{D,T}},
                                 grid_values::AbstractArray{T,D},
                                 grid::UniformGrid{D,T},
                                 basis,
                                 ::Val{S}) where {D, T<:AbstractFloat, S}
    off_range = ntuple(_ -> -(S - 1):S, Val(D))
    inv_h = SVector{D,T}(ntuple(α -> one(T) / grid.spacing[α], Val(D)))

    @inbounds for p in eachindex(positions)
        ξ = particle_to_grid(positions[p], grid)
        m0 = SVector{D,Int}(ntuple(α -> floor(Int, ξ[α]), Val(D)))
        ϕ_per_axis  = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                        ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                          Val(2S)),
                              Val(D))
        ϕp_per_axis = ntuple(α -> ntuple(k -> eval_phi_prime(basis,
                                                              ξ[α] - T(m0[α] + (-(S - 1) + k - 1))),
                                          Val(2S)),
                              Val(D))
        acc = zero(SVector{D,T})
        for off in Iterators.product(off_range...)
            idx = ntuple(α -> m0[α] + off[α], Val(D))
            idx_wrapped, in_bounds = wrap_index(idx, grid)
            in_bounds || continue
            e_m = grid_values[idx_wrapped...]
            acc += _tensor_basis_gradient(ϕ_per_axis, ϕp_per_axis, off, S, inv_h) * e_m
        end
        grads[p] = acc
    end
    return grads
end
