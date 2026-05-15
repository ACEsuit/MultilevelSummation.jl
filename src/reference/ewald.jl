# Naive 3D Ewald reference for Coulomb 1/r in a fully periodic orthorhombic
# cell.
#
# Convention (Gaussian-like, no 4πε₀):
#     U = (1/2) Σ_{i,j,n}'  q_i q_j / |r_i − r_j + n·L|
#
# Ewald decomposition: erfc/erf split with Gaussian smearing exponent α.
#     U = U_real + U_recip + U_self
#     U_real  = (1/2) Σ_{i,j,n}' q_i q_j  erfc(α·s) / s        (s = |r_i−r_j+nL|, s < R_cut)
#     U_recip = (1/(2V)) Σ_{k ≠ 0}  (4π / k²) exp(−k²/(4α²)) |S(k)|²
#             S(k) = Σ_i q_i exp( i k · r_i )
#     U_self  = − (α / √π) Σ_i q_i²

# Public entry points -------------------------------------------------------

"""
    ewald_energy(positions, charges, cell; α, R_cut, k_cut) -> Real

Naive 3D Ewald total energy for fully-periodic orthorhombic Coulomb. The
charge sum must be (approximately) zero.
"""
function ewald_energy(positions::Vector{SVector{3,T}},
                      charges::Vector{T},
                      cell::SMatrix{3,3,T};
                      α::Real, R_cut::Real, k_cut::Real) where {T<:AbstractFloat}
    _assert_orthorhombic(cell)
    _assert_neutral(charges)
    U_real  = _ewald_real_energy(positions, charges, cell, T(α), T(R_cut))
    U_recip = _ewald_recip_energy(positions, charges, cell, T(α), T(k_cut))
    U_self  = -T(α) / sqrt(T(π)) * sum(abs2, charges)
    return U_real + U_recip + U_self
end

"""
    ewald_energy_forces(positions, charges, cell; α, R_cut, k_cut)
        -> (energy::Real, forces::Vector{SVector{3,T}})

Same as `ewald_energy`, but also returns the per-particle forces
`F_i = -∂U/∂r_i`.
"""
function ewald_energy_forces(positions::Vector{SVector{3,T}},
                             charges::Vector{T},
                             cell::SMatrix{3,3,T};
                             α::Real, R_cut::Real, k_cut::Real) where {T<:AbstractFloat}
    _assert_orthorhombic(cell)
    _assert_neutral(charges)
    Ur, Fr = _ewald_real_energy_forces(positions, charges, cell, T(α), T(R_cut))
    Uk, Fk = _ewald_recip_energy_forces(positions, charges, cell, T(α), T(k_cut))
    Us     = -T(α) / sqrt(T(π)) * sum(abs2, charges)
    return Ur + Uk + Us, Fr .+ Fk
end

"""
    ewald_reference(positions, charges, cell; tol = 1e-9) -> Real

Auto-tuned convenience wrapper: pick `(α, R_cut, k_cut)` from the cell
extent and the prescribed truncation tolerance `tol`, then call
[`ewald_energy`](@ref).

Selection rule:

    R_cut = min(box/2 − 0.5, 14)
    α     = √(−log tol) / R_cut
    k_cut = 2 α √(−log tol)

Returns the total Ewald energy. Used by `Tune.run_system_sweep` as the
absolute reference against which MSM sweep results are compared.
"""
function ewald_reference(positions::Vector{SVector{3,T}},
                         charges::Vector{T},
                         cell::SMatrix{3,3,T};
                         tol::Real = 1e-9) where {T<:AbstractFloat}
    box   = cell[1, 1]                       # assumes cubic
    αR    = sqrt(-log(tol))                  # ≈ 4.55 for tol = 1e-9
    R_cut = min(box / 2 - T(0.5), T(14))
    α     = αR / R_cut
    k_cut = 2 * α * αR
    return ewald_energy(positions, charges, cell; α = α, R_cut = R_cut, k_cut = k_cut)
end

# Real-space sum -----------------------------------------------------------

function _ewald_real_energy(positions, charges, cell::SMatrix{3,3,T}, α::T, R_cut::T) where T
    N = length(positions)
    L = SVector{3,T}(cell[1,1], cell[2,2], cell[3,3])
    nmax = ntuple(α_ -> ceil(Int, R_cut / L[α_]), 3)
    R_cut² = R_cut * R_cut
    # Threaded scalar reduction over outer particle index.
    U = tmapreduce(+, 1:N; init = zero(T)) do i
        local_U = zero(T)
        @inbounds for j in 1:N
            qiqj = charges[i] * charges[j]
            for n1 in -nmax[1]:nmax[1], n2 in -nmax[2]:nmax[2], n3 in -nmax[3]:nmax[3]
                (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
                shift = SVector{3,T}(n1 * L[1], n2 * L[2], n3 * L[3])
                r = positions[i] - positions[j] + shift
                s² = sum(abs2, r)
                s² > R_cut² && continue
                s = sqrt(s²)
                local_U += qiqj * erfc(α * s) / s
            end
        end
        local_U
    end
    return U / 2
end

function _ewald_real_energy_forces(positions, charges, cell::SMatrix{3,3,T}, α::T, R_cut::T) where T
    N = length(positions)
    L = SVector{3,T}(cell[1,1], cell[2,2], cell[3,3])
    nmax = ntuple(α_ -> ceil(Int, R_cut / L[α_]), 3)
    F = zeros(SVector{3,T}, N)
    R_cut² = R_cut * R_cut
    inv_sqrt_π = inv(sqrt(T(π)))
    # Thread over outer i; each task writes only F[i].
    U = tmapreduce(+, 1:N; init = zero(T)) do i
        local_U = zero(T)
        local_F = zero(SVector{3,T})
        @inbounds for j in 1:N
            qiqj = charges[i] * charges[j]
            for n1 in -nmax[1]:nmax[1], n2 in -nmax[2]:nmax[2], n3 in -nmax[3]:nmax[3]
                (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
                shift = SVector{3,T}(n1 * L[1], n2 * L[2], n3 * L[3])
                r = positions[i] - positions[j] + shift
                s² = sum(abs2, r)
                s² > R_cut² && continue
                s = sqrt(s²)
                local_U += qiqj * erfc(α * s) / s
                dKds = (-T(2) * α * inv_sqrt_π * exp(-α * α * s²) * s - erfc(α * s)) / s²
                ∇K = (dKds / s) * r
                local_F -= qiqj * ∇K
            end
        end
        @inbounds F[i] = local_F
        local_U
    end
    return U / 2, F
end

# Reciprocal-space sum -----------------------------------------------------
# k = 2π · (m₁/L₁, m₂/L₂, m₃/L₃),  (m₁,m₂,m₃) ∈ ℤ³ \ {0},  |k| ≤ k_cut.

function _ewald_recip_energy(positions, charges, cell::SMatrix{3,3,T}, α::T, k_cut::T) where T
    L = SVector{3,T}(cell[1,1], cell[2,2], cell[3,3])
    V = L[1] * L[2] * L[3]
    mmax = ntuple(α_ -> ceil(Int, k_cut * L[α_] / (2π)), 3)
    k_cut² = k_cut * k_cut
    inv_4α² = inv(T(4) * α * α)
    # Thread over the outer m1 slice; each task handles its own k-plane.
    U = tmapreduce(+, -mmax[1]:mmax[1]; init = zero(T)) do m1
        local_U = zero(T)
        @inbounds for m2 in -mmax[2]:mmax[2], m3 in -mmax[3]:mmax[3]
            (m1 == 0 && m2 == 0 && m3 == 0) && continue
            k = SVector{3,T}(2π * m1 / L[1], 2π * m2 / L[2], 2π * m3 / L[3])
            k² = sum(abs2, k)
            k² > k_cut² && continue
            Sre = zero(T); Sim = zero(T)
            for i in eachindex(positions)
                φ = k[1]*positions[i][1] + k[2]*positions[i][2] + k[3]*positions[i][3]
                Sre += charges[i] * cos(φ)
                Sim += charges[i] * sin(φ)
            end
            local_U += (4π / k²) * exp(-k² * inv_4α²) * (Sre*Sre + Sim*Sim)
        end
        local_U
    end
    return U / (2V)
end

function _ewald_recip_energy_forces(positions, charges, cell::SMatrix{3,3,T}, α::T, k_cut::T) where T
    N = length(positions)
    L = SVector{3,T}(cell[1,1], cell[2,2], cell[3,3])
    V = L[1] * L[2] * L[3]
    mmax = ntuple(α_ -> ceil(Int, k_cut * L[α_] / (2π)), 3)
    k_cut² = k_cut * k_cut
    inv_4α² = inv(T(4) * α * α)

    # Each k-vector contributes to ALL forces F[i], so threading over k needs
    # task-local force arrays. Chunk the outer m1 range, spawn one task per
    # chunk with a local force accumulator, fetch and reduce.
    m1_range = -mmax[1]:mmax[1]
    nchunks  = min(length(m1_range), Threads.nthreads())
    F        = zeros(SVector{3,T}, N)

    function _chunk_kernel(m1_chunk)
        local_U = zero(T)
        local_F = zeros(SVector{3,T}, N)
        @inbounds for m1 in m1_chunk, m2 in -mmax[2]:mmax[2], m3 in -mmax[3]:mmax[3]
            (m1 == 0 && m2 == 0 && m3 == 0) && continue
            k = SVector{3,T}(2π * m1 / L[1], 2π * m2 / L[2], 2π * m3 / L[3])
            k² = sum(abs2, k)
            k² > k_cut² && continue
            Sre = zero(T); Sim = zero(T)
            for i in eachindex(positions)
                φ = k[1]*positions[i][1] + k[2]*positions[i][2] + k[3]*positions[i][3]
                Sre += charges[i] * cos(φ)
                Sim += charges[i] * sin(φ)
            end
            gauss  = exp(-k² * inv_4α²)
            prefac = (4π / k²) * gauss
            local_U += prefac * (Sre*Sre + Sim*Sim)
            for i in 1:N
                φ = k[1]*positions[i][1] + k[2]*positions[i][2] + k[3]*positions[i][3]
                imag_part = cos(φ) * Sim - sin(φ) * Sre
                local_F[i] -= (prefac / V) * charges[i] * imag_part * k
            end
        end
        return local_U, local_F
    end

    if nchunks <= 1
        U_total, F_local = _chunk_kernel(m1_range)
        F .+= F_local
        return U_total / (2V), F
    end

    tasks = map(chunks(m1_range; n = nchunks)) do m1_chunk
        Threads.@spawn _chunk_kernel(m1_chunk)
    end
    U = zero(T)
    for t in tasks
        local_U, local_F = fetch(t)
        U += local_U
        F .+= local_F
    end
    return U / (2V), F
end

# Helpers (Ewald-specific) -------------------------------------------------

# Note: `_assert_orthorhombic` is the shared D-generic helper from
# `src/cell_helpers.jl`, imported into the Reference module.

function _assert_neutral(charges)
    s = sum(charges)
    abs(s) > sqrt(eps(eltype(charges))) * length(charges) &&
        throw(ArgumentError("Ewald reference requires Σ q ≈ 0, got $s"))
end
