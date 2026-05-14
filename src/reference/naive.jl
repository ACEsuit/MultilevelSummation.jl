"""
    naive_energy(positions, charges, cell, periodic, kernel; R_cut=Inf)

Direct `O(N²)` reference energy

    U = ½ Σ_{i,j,n}' q_i · kernel(r_i − r_j + n·L) · q_j

(scalar charge case). The primed sum excludes the term `(i = j, n = 0)`.
Periodic axes are summed over image translations `n` whose shift is
within `R_cut`; non-periodic axes are restricted to `n_α = 0`.

Restricted to **orthorhombic** cells (diagonal lattice matrix).
`R_cut` must be finite when any axis is periodic; for fully open systems
the default `Inf` works.
"""
function naive_energy(positions::AbstractVector{SVector{D,T}},
                      charges::AbstractVector{T},
                      cell::SMatrix{D,D,T},
                      periodic::NTuple{D,Bool},
                      kernel;
                      R_cut::Real = Inf) where {D,T<:AbstractFloat}
    _assert_orthorhombic(cell)
    image_ranges = _image_ranges(cell, periodic, T(R_cut))
    N = length(positions)
    @assert length(charges) == N

    R_cut² = isinf(R_cut) ? T(Inf) : T(R_cut)^2
    # Threaded scalar reduction over the outer particle index. The per-i
    # `local_U` lives inside the closure (task-local) and OhMyThreads
    # merges via `+`. No threadid()-indexed state.
    U = tmapreduce(+, 1:N; init = zero(T)) do i
        local_U = zero(T)
        @inbounds for j in 1:N
            qiqj = charges[i] * charges[j]
            for n in Iterators.product(image_ranges...)
                (i == j && all(==(0), n)) && continue
                shift = _shift(cell, n)
                r     = positions[i] - positions[j] + shift
                sum(abs2, r) > R_cut² && continue
                local_U += qiqj * kernel(r)
            end
        end
        local_U
    end
    return U / 2
end

"""
    naive_energy_forces(positions, charges, cell, periodic, kernel; R_cut=Inf)

Like [`naive_energy`](@ref), but additionally returns the per-particle
forces `F_i = −∂U/∂r_i`. Same `O(N²)` cost and same restrictions
(orthorhombic cell, finite `R_cut` for periodic axes).
"""
function naive_energy_forces(positions::AbstractVector{SVector{D,T}},
                             charges::AbstractVector{T},
                             cell::SMatrix{D,D,T},
                             periodic::NTuple{D,Bool},
                             kernel;
                             R_cut::Real = Inf) where {D,T<:AbstractFloat}
    _assert_orthorhombic(cell)
    image_ranges = _image_ranges(cell, periodic, T(R_cut))
    N = length(positions)
    @assert length(charges) == N

    forces = zeros(SVector{D,T}, N)
    R_cut² = isinf(R_cut) ? T(Inf) : T(R_cut)^2
    # Threaded over outer i. Each task touches only forces[i] for its own
    # i, so the scatter is race-free; energy is gathered via tmapreduce.
    U = tmapreduce(+, 1:N; init = zero(T)) do i
        local_U = zero(T)
        local_F = zero(SVector{D,T})
        @inbounds for j in 1:N
            qiqj = charges[i] * charges[j]
            for n in Iterators.product(image_ranges...)
                (i == j && all(==(0), n)) && continue
                shift = _shift(cell, n)
                r     = positions[i] - positions[j] + shift
                sum(abs2, r) > R_cut² && continue
                local_U += qiqj * kernel(r)
                local_F -= qiqj * grad(kernel, r)
            end
        end
        @inbounds forces[i] = local_F
        local_U
    end
    return U / 2, forces
end

# `_assert_orthorhombic`, `_image_ranges`, `_shift` live in
# `src/cell_helpers.jl` (top-level scope) so `core.jl` can share them.
# They're imported into the Reference module by `Reference.jl`.
