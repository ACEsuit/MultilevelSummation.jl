"""
    UniformGrid{D,T,Per,Sz}(spacing, origin)
    UniformGrid(spacing, size, origin, periodic::NTuple{D,Bool})

A uniform `D`-dimensional grid with per-axis spacing `h_α`, integer
extent `n_α`, an origin offset in particle coordinates, and per-axis
boundary conditions.

**Static type parameters** (compile-time constants):
- `D::Int`           — spatial dimension
- `T<:AbstractFloat` — coordinate floating-point type
- `Per::NTuple{D,Bool}` — `true` = periodic axis, `false` = open
- `Sz::NTuple{D,Int}`   — per-axis grid extent `n_α`

Lifting both `Per` and `Sz` into the type lets the inner loop of every
grid operator dispatch on them and produce fully unrolled, branch-free,
constant-bound code (no `if g.periodic[α]` and no runtime `mod(_, n)`
with `n` from a struct field).

Grid point `m ∈ 0:n_α-1` along axis `α` sits at `origin_α + m·spacing_α`
in particle coordinates. Periodic axes wrap indices `mod n_α`; open axes
drop contributions whose indices fall outside `0:n_α-1`.

`getproperty` shims keep `g.periodic` and `g.size` working as before, so
no call site has to change.
"""
struct UniformGrid{D, T<:AbstractFloat, Per, Sz}
    spacing::SVector{D,T}
    origin::SVector{D,T}

    function UniformGrid{D,T,Per,Sz}(spacing::SVector{D,T},
                                     origin::SVector{D,T}) where {D,T<:AbstractFloat,Per,Sz}
        Per isa NTuple{D,Bool} || throw(ArgumentError(
            "UniformGrid periodicity type parameter must be NTuple{$D,Bool}, got $(typeof(Per))"))
        Sz  isa NTuple{D,Int}  || throw(ArgumentError(
            "UniformGrid size type parameter must be NTuple{$D,Int}, got $(typeof(Sz))"))
        all(>(0), Sz) || throw(ArgumentError("UniformGrid size must be all positive, got $Sz"))
        new{D,T,Per,Sz}(spacing, origin)
    end
end

# Outer constructor: lift runtime tuples (`periodic`, `size`) into the type
# parameters. The new specialisation is created once per (Per, Sz) combination.
UniformGrid(spacing::SVector{D,T}, size::NTuple{D,Int}, origin::SVector{D,T},
            periodic::NTuple{D,Bool}) where {D,T<:AbstractFloat} =
    UniformGrid{D,T,periodic,size}(spacing, origin)

UniformGrid(spacing::NTuple{D,T}, size::NTuple{D,Int}, origin::NTuple{D,T},
            periodic::NTuple{D,Bool}) where {D,T<:AbstractFloat} =
    UniformGrid(SVector{D,T}(spacing...), size, SVector{D,T}(origin...), periodic)

# Backwards-compatible property access: `g.periodic` returns the Per type
# parameter and `g.size` returns the Sz type parameter (both compile-time
# constants), so existing code that reads `g.periodic[α]` or `g.size[α]`
# continues to work without modification — and benefits from constant
# propagation in the caller.
@inline function Base.getproperty(g::UniformGrid{D,T,Per,Sz}, s::Symbol) where {D,T,Per,Sz}
    s === :periodic && return Per
    s === :size     && return Sz
    return getfield(g, s)
end

Base.propertynames(::UniformGrid) = (:spacing, :size, :origin, :periodic)

Base.size(::UniformGrid{D,T,Per,Sz}) where {D,T,Per,Sz}            = Sz
Base.size(::UniformGrid{D,T,Per,Sz}, α::Int) where {D,T,Per,Sz}    = Sz[α]
Base.ndims(::UniformGrid{D}) where D                                = D
spacing(g::UniformGrid)                                             = g.spacing

"Total number of grid points."
npoints(::UniformGrid{D,T,Per,Sz}) where {D,T,Per,Sz} = prod(Sz)

"""
    grid_zeros(T, g)

Allocate a `Array{T, D}` of zeros sized to match `g`.
"""
grid_zeros(::Type{Tv}, ::UniformGrid{D,T,Per,Sz}) where {Tv,D,T,Per,Sz} = zeros(Tv, Sz...)

"""
    wrap_index(idx, g) -> (idx_wrapped::NTuple{D,Int}, in_bounds::Bool)

Map a raw integer index `idx` (0-based, may be negative or out of range)
to a 1-based index suitable for `Array` access. Returns `in_bounds=false`
if the index falls outside an open axis. Periodic axes always wrap.

Implemented as a `@generated` function dispatching on `(Per, Sz)`: each
periodicity-and-size combination compiles to fully unrolled, branch-free
per-axis code with the grid extents inlined as integer literals (so
`mod(idx, n)` becomes `mod(idx, <const>)` and the compiler can replace
the integer division with a multiply-high sequence).
"""
@generated function wrap_index(idx::NTuple{D,Int},
                                ::UniformGrid{D,T,Per,Sz}) where {D,T,Per,Sz}
    @assert Per isa NTuple{D,Bool}
    @assert Sz  isa NTuple{D,Int}
    elem_exprs = Expr[]
    for α in 1:D
        n = Sz[α]
        if Per[α]::Bool
            push!(elem_exprs, :(mod(idx[$α], $n) + 1))
        else
            push!(elem_exprs, :(
                let i = idx[$α]
                    (0 <= i < $n) ? i + 1 : -1
                end
            ))
        end
    end
    if all(Per)
        # All periodic ⇒ in_bounds is statically true; skip the reduction.
        return quote
            $(Expr(:meta, :inline))
            (($(elem_exprs...),), true)
        end
    else
        return quote
            $(Expr(:meta, :inline))
            out = ($(elem_exprs...),)
            (out, all(>(0), out))
        end
    end
end

"""
    particle_to_grid(r, g) -> SVector{D,T}

Convert particle-coordinate position `r` to dimensionless grid coordinates
`ξ_α = (r_α − origin_α) / spacing_α`. The integer part is the nearest
grid point below; the fractional part determines basis weights.
"""
@inline particle_to_grid(r::SVector{D,T}, g::UniformGrid{D,T}) where {D,T} =
    (r - g.origin) ./ g.spacing
