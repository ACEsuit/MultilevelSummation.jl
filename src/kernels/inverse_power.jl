"""
    InversePower{N,T}()

The scalar isotropic pair kernel `K(r) = 1 / |r|^N`. The exponent `N` is
a compile-time type parameter so that `^N` becomes a literal integer
power; `T` is the working floating-point type.

`Coulomb{T}` is provided as the alias `InversePower{1,T}`.

Singular at `r = 0`. Callers must ensure non-coincident particles.
"""
struct InversePower{N,T} end

InversePower{N}() where {N} = InversePower{N,Float64}()
InversePower{N,T}() where {N,T<:AbstractFloat} = (InversePower{N,T}.instance)

const Coulomb{T} = InversePower{1,T}
Coulomb() = Coulomb{Float64}()

# Evaluate the kernel at displacement r.
function (::InversePower{N,T})(r::SVector{D,T}) where {N,D,T<:AbstractFloat}
    s = sqrt(sum(abs2, r))
    return one(T) / s^N
end

# Gradient w.r.t. r:  ∇(|r|^{-N}) = -N |r|^{-N-2} r.
function grad(::InversePower{N,T}, r::SVector{D,T}) where {N,D,T<:AbstractFloat}
    s² = sum(abs2, r)
    s  = sqrt(s²)
    return (-T(N) / s^(N + 2)) * r
end
