"""
    HardyC2Cubic{T}(a::T, L::Int)

Hardy et al. 2015 `C²` cubic softening / splitting matched to the Coulomb
kernel `1/r`. The kernel is decomposed as

    k_0(r) = 1/|r| - (1/a₁) γ(|r|/a₁)                      , a₁ = a
    k_l(r) = (1/aₗ) γ(|r|/aₗ) - (1/aₗ₊₁) γ(|r|/aₗ₊₁)        , l = 1,…,L-1
    k_L(r) = (1/a_L) γ(|r|/a_L)

with `aₗ = 2^{l-1} a` and the `C²` cubic softening

    γ(R) = 1 - (1/2)(R² - 1) + (3/8)(R² - 1)²    for R ≤ 1
    γ(R) = 1/R                                    for R > 1

Valid only when paired with `Coulomb`. Compatibility is verified at
calculator construction; here we just supply the components.
"""
struct HardyC2Cubic{T<:AbstractFloat}
    a::T
    L::Int
    function HardyC2Cubic{T}(a::T, L::Integer) where {T<:AbstractFloat}
        a > 0    || throw(ArgumentError("HardyC2Cubic: a must be > 0, got $a"))
        L >= 1   || throw(ArgumentError("HardyC2Cubic: L must be ≥ 1, got $L"))
        return new{T}(a, Int(L))
    end
end

HardyC2Cubic(a::T, L::Integer) where {T<:AbstractFloat} = HardyC2Cubic{T}(a, L)

requires_neutralising_background(::HardyC2Cubic) = true

"Level spacing aₗ = 2^{l-1} · a, valid for l = 1,…,L."
level_scale(s::HardyC2Cubic{T}, l::Integer) where {T} = T(2)^(l - 1) * s.a

# γ(R): paper softening.
@inline function gamma_softening(R::T) where {T<:AbstractFloat}
    if R <= one(T)
        u = R * R - one(T)
        return one(T) - T(1//2) * u + T(3//8) * u * u
    else
        return one(T) / R
    end
end

# γ'(R) w.r.t. R.
@inline function gamma_softening_prime(R::T) where {T<:AbstractFloat}
    if R <= one(T)
        # γ(R) = 1 - (1/2)(R²-1) + (3/8)(R²-1)²
        # γ'(R) = -R + (3/2)(R²-1) R = R (3/2 R² - 5/2)
        return R * (T(3//2) * R * R - T(5//2))
    else
        return -one(T) / (R * R)
    end
end

# Convenience: f(s) = (1/h) γ(s/h)  and  f'(s) = γ'(s/h) / h².
@inline function _soft_term(s::T, h::T) where {T<:AbstractFloat}
    return gamma_softening(s / h) / h
end

# d/dr_α [ (1/h) γ(|r|/h) ] = γ'(|r|/h) · r_α / (h² |r|).
@inline function _soft_term_grad(r::SVector{D,T}, h::T) where {D,T<:AbstractFloat}
    s = sqrt(sum(abs2, r))
    R = s / h
    coeff = gamma_softening_prime(R) / (h * h * s)
    return coeff * r
end

# --- splitting interface --------------------------------------------------

# K_0(r) = 1/|r| - (1/a) γ(|r|/a),  supported in |r| ≤ a.
function short_range(s::HardyC2Cubic{T}, r::SVector{D,T}) where {D,T<:AbstractFloat}
    sr = sqrt(sum(abs2, r))
    return one(T) / sr - _soft_term(sr, s.a)
end

function short_range_grad(s::HardyC2Cubic{T}, r::SVector{D,T}) where {D,T<:AbstractFloat}
    sr  = sqrt(sum(abs2, r))
    # ∇(1/|r|) = -r/|r|³
    return (-one(T) / sr^3) * r .- _soft_term_grad(r, s.a)
end

# K_l(r) for l = 1, …, L-1.
function long_range_level(s::HardyC2Cubic{T}, l::Integer, r::SVector{D,T}) where {D,T<:AbstractFloat}
    1 <= l <= s.L - 1 || throw(ArgumentError("level l must be in 1:L-1 = 1:$(s.L-1), got $l"))
    sr  = sqrt(sum(abs2, r))
    h_l   = level_scale(s, l)       # 2^{l-1} a
    h_lp1 = level_scale(s, l + 1)   # 2^l a
    return _soft_term(sr, h_l) - _soft_term(sr, h_lp1)
end

function long_range_level_grad(s::HardyC2Cubic{T}, l::Integer, r::SVector{D,T}) where {D,T<:AbstractFloat}
    1 <= l <= s.L - 1 || throw(ArgumentError("level l must be in 1:L-1 = 1:$(s.L-1), got $l"))
    h_l   = level_scale(s, l)
    h_lp1 = level_scale(s, l + 1)
    return _soft_term_grad(r, h_l) .- _soft_term_grad(r, h_lp1)
end

# K_L(r) = (1/a_L) γ(|r|/a_L).
function top_level(s::HardyC2Cubic{T}, r::SVector{D,T}) where {D,T<:AbstractFloat}
    sr  = sqrt(sum(abs2, r))
    h_L = level_scale(s, s.L)
    return _soft_term(sr, h_L)
end

function top_level_grad(s::HardyC2Cubic{T}, r::SVector{D,T}) where {D,T<:AbstractFloat}
    h_L = level_scale(s, s.L)
    return _soft_term_grad(r, h_L)
end
