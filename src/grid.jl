"""
    UniformGrid{D,T}(spacing, size, origin, periodic)

A uniform `D`-dimensional grid with per-axis spacing `h_α`, integer
extent `n_α`, an origin offset in particle coordinates, and per-axis
boundary conditions (`:periodic`/`:open` encoded as a Bool).

Grid point `m ∈ 0:n_α-1` along axis `α` sits at `origin_α + m·spacing_α`
in particle coordinates.

Periodic axes wrap indices `mod n_α`; open axes drop contributions whose
indices fall outside `0:n_α-1`.
"""
struct UniformGrid{D,T<:AbstractFloat}
    spacing::SVector{D,T}
    size::NTuple{D,Int}
    origin::SVector{D,T}
    periodic::NTuple{D,Bool}
end

# Convenience constructors
UniformGrid(spacing::NTuple{D,T}, size::NTuple{D,Int}, origin::NTuple{D,T},
            periodic::NTuple{D,Bool}) where {D,T<:AbstractFloat} =
    UniformGrid{D,T}(SVector{D,T}(spacing...), size, SVector{D,T}(origin...), periodic)

Base.size(g::UniformGrid)            = g.size
Base.size(g::UniformGrid, α::Int)    = g.size[α]
Base.ndims(::UniformGrid{D}) where D = D
spacing(g::UniformGrid)              = g.spacing

"Total number of grid points."
npoints(g::UniformGrid)              = prod(g.size)

"""
    grid_zeros(T, g)

Allocate a `Array{T, D}` of zeros sized to match `g`.
"""
grid_zeros(::Type{T}, g::UniformGrid{D}) where {T,D} = zeros(T, g.size...)

"""
    wrap_index(idx, g) -> (idx_wrapped::NTuple{D,Int}, in_bounds::Bool)

Map a raw integer index `idx` (0-based, may be negative or out of range)
to a 1-based index suitable for `Array` access. Returns `in_bounds=false`
if the index falls outside an open axis. Periodic axes always wrap.
"""
@inline function wrap_index(idx::NTuple{D,Int}, g::UniformGrid{D}) where {D}
    out = ntuple(Val(D)) do α
        if g.periodic[α]
            mod(idx[α], g.size[α]) + 1     # 1-based
        else
            (0 <= idx[α] < g.size[α]) ? idx[α] + 1 : -1
        end
    end
    in_bounds = all(>(0), out)
    return out, in_bounds
end

"""
    particle_to_grid(r, g) -> SVector{D,T}

Convert particle-coordinate position `r` to dimensionless grid coordinates
`ξ_α = (r_α − origin_α) / spacing_α`. The integer part is the nearest
grid point below; the fractional part determines basis weights.
"""
@inline particle_to_grid(r::SVector{D,T}, g::UniformGrid{D,T}) where {D,T} =
    (r - g.origin) ./ g.spacing
