"""
    MLSumCalculator{T,S,B}(splitting, basis, h; charge_property=:charge)

Container for MSM hyperparameters at the low-level (numerical) layer; also
serves as the `AtomsCalculators.AbstractCalculator` for AtomsBase systems.

Fields:
- `splitting`: a kernel splitting (e.g. `HardyC2Cubic{T}`), carrying `a` and `L`.
- `basis`: an interpolation basis (e.g. `CubicC1{T}`).
- `h::T`: the finest-level grid spacing (same along every axis, prototype).
- `charge_property::Symbol`: which AtomsBase atom property to read for charges
  when used through the AtomsCalculators interface (default `:charge`).
"""
struct MLSumCalculator{T<:AbstractFloat, S, B}
    splitting::S
    basis::B
    h::T
    charge_property::Symbol
end

MLSumCalculator(splitting::S, basis::B, h::T;
                charge_property::Symbol = :charge) where {T<:AbstractFloat,S,B} =
    MLSumCalculator{T,S,B}(splitting, basis, h, charge_property)

# Convenience accessors
short_range_cutoff(c::MLSumCalculator)    = c.splitting.a
nlevels(c::MLSumCalculator)               = c.splitting.L
fine_spacing(c::MLSumCalculator)          = c.h

"""
    kernel_self_value(splitting, ::Val{D}) -> T

Value of the total long-range kernel at zero separation,
`k_long(0) = Σ_{l=1}^{L} k_l(0)`. Used to subtract the spurious
self-interaction picked up by the long-range pathway (paper eq. 5,
correction term with `j = i`).

Computed by directly summing the splitting components at `r = 0`.
"""
function kernel_self_value(splitting, ::Val{D}, ::Type{T}) where {D, T<:AbstractFloat}
    z = zero(SVector{D,T})
    v = top_level(splitting, z)
    for l in 1:(splitting.L - 1)
        v += long_range_level(splitting, l, z)
    end
    return v
end

# --- Grid hierarchy ----------------------------------------------------------

"""
    build_grid_hierarchy(cell, periodic, positions, c) -> Vector{UniformGrid{D,T}}

Build the finest-to-coarsest MSM grid hierarchy. For periodic axes, the
level-1 extent is `L_α / h` (must be an integer multiple of `2^{L-1}`).
For open axes, the level-1 extent is set from the particle bounding box
plus a basis-support pad, rounded up to a multiple of `2^{L-1}`.

Mixed BC: any combination is allowed; each axis is treated independently.
"""
function build_grid_hierarchy(cell::SMatrix{D,D,T},
                              periodic::NTuple{D,Bool},
                              positions::AbstractVector{SVector{D,T}},
                              c::MLSumCalculator{T}) where {D, T<:AbstractFloat}
    h    = c.h
    L    = nlevels(c)
    twoL = 2^(L - 1)
    pad  = T(support_radius(c.basis)) * h    # basis half-width in particle coords

    n1 = ntuple(Val(D)) do α
        if periodic[α]
            n = Int(round(cell[α, α] / h))
            n > 0 || throw(ArgumentError("periodic axis $α: derived n=$n ≤ 0"))
            rem(n, twoL) == 0 ||
                throw(ArgumentError("periodic axis $α: L_α/h = $n not divisible by $twoL = 2^(L-1)"))
            n
        else
            lo  = minimum(p -> p[α], positions) - pad
            hi  = maximum(p -> p[α], positions) + pad
            n   = max(ceil(Int, (hi - lo) / h), 1)
            ((n + twoL - 1) ÷ twoL) * twoL
        end
    end

    origin = SVector{D,T}(ntuple(Val(D)) do α
        if periodic[α]
            zero(T)
        else
            minimum(p -> p[α], positions) - pad
        end
    end)

    spacing = SVector{D,T}(ntuple(_ -> h, Val(D)))
    g_fine = UniformGrid{D,T}(spacing, n1, origin, periodic)

    grids = Vector{UniformGrid{D,T}}(undef, L)
    grids[1] = g_fine
    for l in 2:L
        grids[l] = coarser_grid(grids[l-1], 2)
    end
    return grids
end

# --- End-to-end MSM ---------------------------------------------------------

"""
    msm_energy(positions, charges, cell, periodic, calc) -> energy::T

Compute the MSM-approximated energy
    U^MSM = ½ Σ_{i≠j} q_i q_j K(r_ij)
for charges `q_i` at positions `r_i` in an orthorhombic `cell` with per-axis
`periodic` flags, using the calculator `calc`.

Only `j = i` is excluded (no bonded-pair exclusions in the prototype).
"""
function msm_energy(positions::AbstractVector{SVector{D,T}},
                    charges::AbstractVector{T},
                    cell::SMatrix{D,D,T},
                    periodic::NTuple{D,Bool},
                    calc::MLSumCalculator{T}) where {D, T<:AbstractFloat}
    U, _ = _msm_compute(positions, charges, cell, periodic, calc; want_forces=false)
    return U
end

"""
    msm_energy_forces(positions, charges, cell, periodic, calc) -> (energy, forces)

Compute MSM energy and analytic per-particle forces.
"""
function msm_energy_forces(positions::AbstractVector{SVector{D,T}},
                           charges::AbstractVector{T},
                           cell::SMatrix{D,D,T},
                           periodic::NTuple{D,Bool},
                           calc::MLSumCalculator{T}) where {D, T<:AbstractFloat}
    return _msm_compute(positions, charges, cell, periodic, calc; want_forces=true)
end

function _msm_compute(positions::AbstractVector{SVector{D,T}},
                      charges::AbstractVector{T},
                      cell::SMatrix{D,D,T},
                      periodic::NTuple{D,Bool},
                      calc::MLSumCalculator{T};
                      want_forces::Bool) where {D, T<:AbstractFloat}
    @assert length(positions) == length(charges)
    N = length(positions)
    splitting = calc.splitting
    basis     = calc.basis
    L         = nlevels(calc)

    # --- Build hierarchy and intermediate fields ---------------------------
    grids = build_grid_hierarchy(cell, periodic, positions, calc)
    qs = [zeros(T, g.size...) for g in grids]
    es = [zeros(T, g.size...) for g in grids]

    # 1. Anterpolate particles to the finest grid
    anterpolate!(qs[1], positions, charges, grids[1], basis)

    # 2. Restrict q^1 → q^2 → ... → q^L
    for l in 1:(L - 1)
        restrict!(qs[l+1], qs[l], grids[l+1], grids[l], basis)
    end

    # 3. Neutralising background at the top, if applicable
    if requires_neutralising_background(splitting) && all(periodic)
        apply_neutralising_background!(qs[end], splitting, grids[end])
    end

    # 4. Grid-cutoff convolutions for l = 1, …, L-1
    for l in 1:(L - 1)
        grid_cutoff!(es[l], qs[l], grids[l], splitting, l)
    end

    # 5. Top-level direct sum
    top_level!(es[end], qs[end], grids[end], splitting)

    # 6. Prolong potentials downward: e^l += prolong(e^{l+1})
    for l in (L - 1):-1:1
        tmp = zeros(T, grids[l].size...)
        prolong!(tmp, es[l+1], grids[l], grids[l+1], basis)
        @inbounds @. es[l] += tmp
    end

    # 7. Interpolate e^1 to particle potentials
    pots = zeros(T, N)
    interpolate!(pots, positions, es[1], grids[1], basis)

    # 8. Long-range energy
    U_long = T(1//2) * sum(charges[i] * pots[i] for i in 1:N)

    # 9. Self-interaction correction (subtracts the j=i self picked up by U_long)
    k0_self = kernel_self_value(splitting, Val(D), T)
    U_self  = -T(1//2) * k0_self * sum(abs2, charges)

    # 10. Short-range direct sum  (no exclusions other than j=i)
    a = short_range_cutoff(calc)
    U_short, F_short, F_long_arr = _short_range_and_long_forces(
        positions, charges, cell, periodic, splitting, basis, grids, es, calc;
        want_forces=want_forces, a=a,
    )

    U_total = U_short + U_long + U_self

    if want_forces
        return U_total, F_short .+ F_long_arr
    else
        return U_total, Vector{SVector{D,T}}(undef, 0)
    end
end

# Compute short-range energy/forces and (if requested) the long-range forces
# from the interpolated grid potential gradient.
function _short_range_and_long_forces(positions::AbstractVector{SVector{D,T}},
                                      charges::AbstractVector{T},
                                      cell::SMatrix{D,D,T},
                                      periodic::NTuple{D,Bool},
                                      splitting,
                                      basis,
                                      grids,
                                      es,
                                      calc::MLSumCalculator{T};
                                      want_forces::Bool,
                                      a::T) where {D, T<:AbstractFloat}
    N = length(positions)
    image_ranges = _image_ranges(cell, periodic, a)
    a² = a * a

    U_short = zero(T)
    F_short = zeros(SVector{D,T}, N)
    @inbounds for i in 1:N, j in 1:N
        qiqj = charges[i] * charges[j]
        for n in Iterators.product(image_ranges...)
            (i == j && all(==(0), n)) && continue
            shift = _shift(cell, n)
            r     = positions[i] - positions[j] + shift
            sum(abs2, r) > a² && continue
            U_short += qiqj * short_range(splitting, r)
            if want_forces
                F_short[i] -= qiqj * short_range_grad(splitting, r)
            end
        end
    end
    U_short *= T(1//2)

    # Long-range force: F_i_long = -q_i · interpolate_grad(e^1) at r_i
    F_long = zeros(SVector{D,T}, N)
    if want_forces
        interp_grads = zeros(SVector{D,T}, N)
        interpolate_grad!(interp_grads, positions, es[1], grids[1], basis)
        @inbounds for i in 1:N
            F_long[i] = -charges[i] * interp_grads[i]
        end
    end

    return U_short, F_short, F_long
end
