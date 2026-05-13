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
function anterpolate!(grid_values::AbstractArray{T,D},
                      positions::AbstractVector{SVector{D,T}},
                      charges::AbstractVector{T},
                      grid::UniformGrid{D,T},
                      basis) where {D,T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(positions) == length(charges)
    fill!(grid_values, zero(T))
    s = support_radius(basis)
    # Per-axis offset range: from -(s-1) to s gives 2s integer m values
    # spanning the full Φ support [−s, s) around the floor of ξ.
    off_range = ntuple(_ -> -(s - 1):s, Val(D))

    @inbounds for p in eachindex(positions)
        ξ = particle_to_grid(positions[p], grid)
        q = charges[p]
        m0 = SVector{D,Int}(ntuple(α -> floor(Int, ξ[α]), Val(D)))
        # Precompute per-axis basis values for the window.
        ϕ_per_axis = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                      ξ[α] - T(m0[α] + (-(s - 1) + k - 1))),
                                        Val(2s)),
                            Val(D))
        for off in Iterators.product(off_range...)
            idx = ntuple(α -> m0[α] + off[α], Val(D))
            idx_wrapped, in_bounds = wrap_index(idx, grid)
            in_bounds || continue
            # tensor-product basis value
            bv = one(T)
            for α in 1:D
                k = off[α] - (-(s - 1)) + 1
                bv *= ϕ_per_axis[α][k]
            end
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
function interpolate!(potentials::AbstractVector{T},
                      positions::AbstractVector{SVector{D,T}},
                      grid_values::AbstractArray{T,D},
                      grid::UniformGrid{D,T},
                      basis) where {D,T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(potentials) == length(positions)
    fill!(potentials, zero(T))
    s = support_radius(basis)
    off_range = ntuple(_ -> -(s - 1):s, Val(D))

    @inbounds for p in eachindex(positions)
        ξ = particle_to_grid(positions[p], grid)
        m0 = SVector{D,Int}(ntuple(α -> floor(Int, ξ[α]), Val(D)))
        ϕ_per_axis = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                      ξ[α] - T(m0[α] + (-(s - 1) + k - 1))),
                                        Val(2s)),
                            Val(D))
        acc = zero(T)
        for off in Iterators.product(off_range...)
            idx = ntuple(α -> m0[α] + off[α], Val(D))
            idx_wrapped, in_bounds = wrap_index(idx, grid)
            in_bounds || continue
            bv = one(T)
            for α in 1:D
                k = off[α] - (-(s - 1)) + 1
                bv *= ϕ_per_axis[α][k]
            end
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
function interpolate_grad!(grads::AbstractVector{SVector{D,T}},
                           positions::AbstractVector{SVector{D,T}},
                           grid_values::AbstractArray{T,D},
                           grid::UniformGrid{D,T},
                           basis) where {D,T<:AbstractFloat}
    @assert size(grid_values) == grid.size
    @assert length(grads) == length(positions)
    fill!(grads, zero(SVector{D,T}))
    s = support_radius(basis)
    off_range = ntuple(_ -> -(s - 1):s, Val(D))
    inv_h = SVector{D,T}(ntuple(α -> one(T) / grid.spacing[α], Val(D)))

    @inbounds for p in eachindex(positions)
        ξ = particle_to_grid(positions[p], grid)
        m0 = SVector{D,Int}(ntuple(α -> floor(Int, ξ[α]), Val(D)))
        ϕ_per_axis  = ntuple(α -> ntuple(k -> eval_phi(basis,
                                                        ξ[α] - T(m0[α] + (-(s - 1) + k - 1))),
                                          Val(2s)),
                              Val(D))
        ϕp_per_axis = ntuple(α -> ntuple(k -> eval_phi_prime(basis,
                                                              ξ[α] - T(m0[α] + (-(s - 1) + k - 1))),
                                          Val(2s)),
                              Val(D))
        acc = zero(SVector{D,T})
        for off in Iterators.product(off_range...)
            idx = ntuple(α -> m0[α] + off[α], Val(D))
            idx_wrapped, in_bounds = wrap_index(idx, grid)
            in_bounds || continue
            e_m = grid_values[idx_wrapped...]
            # gradient w.r.t. ξ_α of the tensor product; chain to r via /h
            for α in 1:D
                g_factor = one(T)
                for β in 1:D
                    k = off[β] - (-(s - 1)) + 1
                    g_factor *= (β == α ? ϕp_per_axis[β][k] : ϕ_per_axis[β][k])
                end
                acc = setindex(acc, acc[α] + g_factor * e_m * inv_h[α], α)
            end
        end
        grads[p] = acc
    end
    return grads
end
