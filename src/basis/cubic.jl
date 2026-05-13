"""
    CubicC1{T}()

Piecewise-cubic interpolation basis with `C¹` continuity, support
`|ξ| ≤ 2`, from Hardy et al. 2015 (paper, just after eq. 13):

    Φ(ξ) = (1 - |ξ|)·(1 + |ξ| - (3/2) ξ²)   for 0 ≤ |ξ| ≤ 1
         = -(1/2)·(|ξ| - 1)·(2 - |ξ|)²       for 1 ≤ |ξ| ≤ 2
         = 0                                  otherwise
"""
struct CubicC1{T<:AbstractFloat} end
CubicC1() = CubicC1{Float64}()

"Support half-width in grid spacings (cubic ⇒ 2)."
support_radius(::CubicC1) = 2

function eval_phi(::CubicC1{T}, ξ::T) where {T<:AbstractFloat}
    a = abs(ξ)
    if a <= one(T)
        return (one(T) - a) * (one(T) + a - T(3//2) * a * a)
    elseif a <= T(2)
        return -T(1//2) * (a - one(T)) * (T(2) - a)^2
    else
        return zero(T)
    end
end

# dΦ/dξ. Sign follows from chain-rule on |ξ|: d|ξ|/dξ = sign(ξ).
function eval_phi_prime(::CubicC1{T}, ξ::T) where {T<:AbstractFloat}
    a = abs(ξ)
    if a <= one(T)
        # d/da[ (1-a)(1+a-3/2 a²) ] = -(1+a-3/2 a²) + (1-a)(1 - 3a)
        dΦda = -(one(T) + a - T(3//2) * a * a) + (one(T) - a) * (one(T) - T(3) * a)
    elseif a <= T(2)
        # d/da[ -1/2 (a-1)(2-a)^2 ] = -1/2 [ (2-a)^2 + (a-1)·2(2-a)·(-1) ]
        #                            = -1/2 (2-a) [ (2-a) - 2(a-1) ]
        #                            = -1/2 (2-a)(4 - 3a)
        dΦda = -T(1//2) * (T(2) - a) * (T(4) - T(3) * a)
    else
        return zero(T)
    end
    return sign(ξ) * dΦda
end
