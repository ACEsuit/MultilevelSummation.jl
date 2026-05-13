"""
    RationalDecay{N,T}(r₀)

The scalar isotropic pair kernel `K(r) = 1 / (1 + (|r|/r₀)^N)`.

Smooth at the origin (equals `1` there), decays like `(r₀/|r|)^N` for
`|r| ≫ r₀`. `N` is a compile-time type parameter; `r₀::T` sets the
transition length.
"""
struct RationalDecay{N,T<:AbstractFloat}
    r₀::T
end

RationalDecay{N}(r₀::T) where {N,T<:AbstractFloat} = RationalDecay{N,T}(r₀)

function (K::RationalDecay{N,T})(r::SVector{D,T}) where {N,D,T<:AbstractFloat}
    s² = sum(abs2, r)
    s  = sqrt(s²)
    u  = (s / K.r₀)^N
    return one(T) / (one(T) + u)
end

# ∇K = -N (s/r₀)^{N-1} (1/r₀) / (1 + u)^2 · (r/s)
#    = -N s^{N-2} / r₀^N / (1+u)^2 · r
function grad(K::RationalDecay{N,T}, r::SVector{D,T}) where {N,D,T<:AbstractFloat}
    s²    = sum(abs2, r)
    s     = sqrt(s²)
    u     = (s / K.r₀)^N
    denom = (one(T) + u)^2
    factor = -T(N) * s^(N - 2) / (K.r₀^N * denom)
    return factor * r
end
