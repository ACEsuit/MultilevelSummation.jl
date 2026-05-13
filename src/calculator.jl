using AtomsBase: AbstractSystem, position, cell_vectors, periodicity, n_dimensions
using Unitful: ustrip
import AtomsCalculators

"""
    _strip_to_core(sys, calc) -> (positions, charges, cell, periodic)

Pull the data the numerical core needs out of an AtomsBase system, stripping
Unitful units to plain `T`s.

**Unit convention**: this prototype is unit-agnostic. The user is responsible
for passing a system whose positions and the calculator's `h`, `a` are in the
same length unit; charges must be dimensionless (or convertible to
dimensionless via `ustrip`). The numerical core knows nothing about units.

The charge property name is read from `calc.charge_property` (default
`:charge`).
"""
function _strip_to_core(sys::AbstractSystem,
                        calc::MSMCalculator{T}) where {T<:AbstractFloat}
    D = n_dimensions(sys)
    N = length(sys)
    pers = periodicity(sys)
    @assert length(pers) == D

    # Positions (D-vectors, units stripped)
    positions = Vector{SVector{D,T}}(undef, N)
    @inbounds for i in 1:N
        p = position(sys, i)
        positions[i] = SVector{D,T}(ntuple(α -> T(ustrip(p[α])), D))
    end

    # Charges (scalar, unit stripped if necessary)
    charges = Vector{T}(undef, N)
    @inbounds for i in 1:N
        q = sys[i, calc.charge_property]
        charges[i] = T(ustrip(q))
    end

    # Cell: lattice vectors as columns of an SMatrix{D,D,T}
    cv = cell_vectors(sys)
    cell = SMatrix{D,D,T}(reduce(hcat,
        SVector{D,T}(ntuple(α -> T(ustrip(cv[β][α])), D)) for β in 1:D))

    return positions, charges, cell, NTuple{D,Bool}(pers)
end

# --- AtomsCalculators interface ----------------------------------------------
# We extend the generic methods to accept an MSMCalculator. They convert the
# AtomsBase system to raw arrays and delegate to the numerical core.

function AtomsCalculators.potential_energy(sys::AbstractSystem,
                                            calc::MSMCalculator;
                                            kwargs...)
    positions, charges, cell, periodic = _strip_to_core(sys, calc)
    return msm_energy(positions, charges, cell, periodic, calc)
end

function AtomsCalculators.forces(sys::AbstractSystem,
                                  calc::MSMCalculator;
                                  kwargs...)
    positions, charges, cell, periodic = _strip_to_core(sys, calc)
    _, F = msm_energy_forces(positions, charges, cell, periodic, calc)
    return F
end

function AtomsCalculators.forces!(F::AbstractVector{<:SVector{D,T}},
                                   sys::AbstractSystem,
                                   calc::MSMCalculator{T};
                                   kwargs...) where {D,T}
    positions, charges, cell, periodic = _strip_to_core(sys, calc)
    _, Fout = msm_energy_forces(positions, charges, cell, periodic, calc)
    @assert length(F) == length(Fout)
    @inbounds for i in eachindex(F)
        F[i] = Fout[i]
    end
    return F
end

function AtomsCalculators.energy_forces(sys::AbstractSystem,
                                         calc::MSMCalculator;
                                         kwargs...)
    positions, charges, cell, periodic = _strip_to_core(sys, calc)
    U, F = msm_energy_forces(positions, charges, cell, periodic, calc)
    return (energy = U, forces = F)
end
