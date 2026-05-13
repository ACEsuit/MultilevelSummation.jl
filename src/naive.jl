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

    U      = zero(T)
    R_cut² = isinf(R_cut) ? T(Inf) : T(R_cut)^2
    @inbounds for i in 1:N, j in 1:N
        qiqj = charges[i] * charges[j]
        for n in Iterators.product(image_ranges...)
            (i == j && all(==(0), n)) && continue
            shift = _shift(cell, n)
            r     = positions[i] - positions[j] + shift
            sum(abs2, r) > R_cut² && continue
            U += qiqj * kernel(r)
        end
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

    U      = zero(T)
    forces = zeros(SVector{D,T}, N)
    R_cut² = isinf(R_cut) ? T(Inf) : T(R_cut)^2
    @inbounds for i in 1:N, j in 1:N
        qiqj = charges[i] * charges[j]
        for n in Iterators.product(image_ranges...)
            (i == j && all(==(0), n)) && continue
            shift = _shift(cell, n)
            r     = positions[i] - positions[j] + shift
            sum(abs2, r) > R_cut² && continue
            U          += qiqj * kernel(r)
            forces[i]  -= qiqj * grad(kernel, r)
        end
    end
    return U / 2, forces
end

# --- helpers ---------------------------------------------------------------

function _assert_orthorhombic(cell::SMatrix{D,D,T}) where {D,T}
    @inbounds for α in 1:D, β in 1:D
        if α != β && !iszero(cell[α, β])
            throw(ArgumentError(
                "naive reference only supports orthorhombic cells; got cell[$α,$β] = $(cell[α, β])"
            ))
        end
    end
    return nothing
end

function _image_ranges(cell::SMatrix{D,D,T}, periodic::NTuple{D,Bool},
                       R_cut::T) where {D,T}
    return ntuple(D) do α
        if periodic[α]
            L = cell[α, α]
            L > 0 || throw(ArgumentError("periodic axis $α has L = $L ≤ 0"))
            isfinite(R_cut) || throw(ArgumentError(
                "R_cut must be finite for periodic axes (got $R_cut)"
            ))
            n_max = ceil(Int, R_cut / L)
            -n_max:n_max
        else
            0:0
        end
    end
end

@inline function _shift(cell::SMatrix{D,D,T}, n::NTuple{D,Int}) where {D,T}
    # Orthorhombic only: shift = Σ_α n_α · L_α e_α
    return SVector{D,T}(ntuple(α -> T(n[α]) * cell[α, α], D)...)
end
