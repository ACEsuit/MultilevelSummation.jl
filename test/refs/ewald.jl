module EwaldRef
# Naive 3D Ewald reference for Coulomb 1/r in a fully periodic orthorhombic
# cell. Lives only in `test/` — not part of the shipped package.
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

using StaticArrays, SpecialFunctions

# Public entry points -------------------------------------------------------

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

# Real-space sum -----------------------------------------------------------

function _ewald_real_energy(positions, charges, cell::SMatrix{3,3,T}, α::T, R_cut::T) where T
    N = length(positions)
    L = SVector{3,T}(cell[1,1], cell[2,2], cell[3,3])
    nmax = ntuple(α_ -> ceil(Int, R_cut / L[α_]), 3)
    U = zero(T)
    R_cut² = R_cut * R_cut
    @inbounds for i in 1:N, j in 1:N
        qiqj = charges[i] * charges[j]
        for n1 in -nmax[1]:nmax[1], n2 in -nmax[2]:nmax[2], n3 in -nmax[3]:nmax[3]
            (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
            shift = SVector{3,T}(n1 * L[1], n2 * L[2], n3 * L[3])
            r = positions[i] - positions[j] + shift
            s² = sum(abs2, r)
            s² > R_cut² && continue
            s = sqrt(s²)
            U += qiqj * erfc(α * s) / s
        end
    end
    return U / 2
end

function _ewald_real_energy_forces(positions, charges, cell::SMatrix{3,3,T}, α::T, R_cut::T) where T
    N = length(positions)
    L = SVector{3,T}(cell[1,1], cell[2,2], cell[3,3])
    nmax = ntuple(α_ -> ceil(Int, R_cut / L[α_]), 3)
    U = zero(T)
    F = zeros(SVector{3,T}, N)
    R_cut² = R_cut * R_cut
    inv_sqrt_π = inv(sqrt(T(π)))
    @inbounds for i in 1:N, j in 1:N
        qiqj = charges[i] * charges[j]
        for n1 in -nmax[1]:nmax[1], n2 in -nmax[2]:nmax[2], n3 in -nmax[3]:nmax[3]
            (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
            shift = SVector{3,T}(n1 * L[1], n2 * L[2], n3 * L[3])
            r = positions[i] - positions[j] + shift
            s² = sum(abs2, r)
            s² > R_cut² && continue
            s = sqrt(s²)
            U += qiqj * erfc(α * s) / s
            # ∇K_real(r) where K_real(s) = erfc(α s)/s.
            # dK/ds = [−2α/√π · exp(−α² s²) · s − erfc(α s)] / s²
            dKds = (-T(2) * α * inv_sqrt_π * exp(-α * α * s²) * s - erfc(α * s)) / s²
            ∇K = (dKds / s) * r                       # = (dK/ds)·r̂
            F[i] -= qiqj * ∇K
        end
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
    U = zero(T)
    @inbounds for m1 in -mmax[1]:mmax[1], m2 in -mmax[2]:mmax[2], m3 in -mmax[3]:mmax[3]
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
        U += (4π / k²) * exp(-k² * inv_4α²) * (Sre*Sre + Sim*Sim)
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
    U = zero(T)
    F = zeros(SVector{3,T}, N)
    @inbounds for m1 in -mmax[1]:mmax[1], m2 in -mmax[2]:mmax[2], m3 in -mmax[3]:mmax[3]
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
        gauss = exp(-k² * inv_4α²)
        prefac = (4π / k²) * gauss
        U += prefac * (Sre*Sre + Sim*Sim)
        # ∂|S(k)|²/∂r_iα = 2 q_i k_α · (Sim cos(k·r_i) − Sre sin(k·r_i))
        # F_iα = -∂U_recip/∂r_iα = -(prefac / V) · q_i · k_α · (Sim cos − Sre sin)
        for i in 1:N
            φ = k[1]*positions[i][1] + k[2]*positions[i][2] + k[3]*positions[i][3]
            imag_part = cos(φ) * Sim - sin(φ) * Sre
            F[i] -= (prefac / V) * charges[i] * imag_part * k
        end
    end
    return U / (2V), F
end

# Helpers ------------------------------------------------------------------

function _assert_orthorhombic(cell::SMatrix{3,3,T}) where T
    for α in 1:3, β in 1:3
        α != β && !iszero(cell[α, β]) &&
            throw(ArgumentError("Ewald reference requires orthorhombic cell"))
    end
end

function _assert_neutral(charges)
    s = sum(charges)
    abs(s) > sqrt(eps(eltype(charges))) * length(charges) &&
        throw(ArgumentError("Ewald reference requires Σ q ≈ 0, got $s"))
end

end # module EwaldRef
