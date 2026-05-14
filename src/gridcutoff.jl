"""
    build_stencil(splitting, l, h_grid) -> (stencil, smax)

Precompute the level-`l` grid-cutoff stencil for the given `splitting`
at grid spacing `h_grid`. Returns a `(2·smax+1)`-shaped tensor of stencil
values together with the per-axis half-width `smax`.

The level-`l` "long-range" kernel `k_l(r)` (`1 ≤ l ≤ L-1`) has compact
support `|r| ≤ a_{l+1} = 2^l · a`. The stencil radius is `smax_α =
ceil(a_{l+1} / h_grid_α)`.
"""
function build_stencil(splitting, l::Int, h_grid::SVector{D,T}) where {D,T<:AbstractFloat}
    a_lp1 = T(2)^l * splitting.a
    smax  = ntuple(α -> ceil(Int, a_lp1 / h_grid[α]), Val(D))
    dims  = ntuple(α -> 2 * smax[α] + 1, Val(D))
    stencil = zeros(T, dims...)
    @inbounds for I in CartesianIndices(stencil)
        off = ntuple(α -> I[α] - 1 - smax[α], Val(D))
        Δr  = SVector{D,T}(ntuple(α -> T(off[α]) * h_grid[α], Val(D)))
        stencil[I] = long_range_level(splitting, l, Δr)
    end
    return stencil, smax
end

"""
    grid_cutoff!(e, q, grid, splitting, l) -> e

Compute the level-`l` "grid cutoff" potential (paper eq. 9):

    e[m] = Σ_n k_l(r_m − r_n) q[n]

where the sum is over destination grid points `n` within the compact
support of `k_l`. Periodic axes wrap; open axes drop out-of-bounds
contributions.

`e` is overwritten.
"""
function grid_cutoff!(e::AbstractArray{T,D},
                      q::AbstractArray{T,D},
                      grid::UniformGrid{D,T},
                      splitting,
                      l::Int) where {D,T<:AbstractFloat}
    @assert size(e) == size(q) == grid.size
    stencil, smax = build_stencil(splitting, l, grid.spacing)
    return _convolve!(e, q, stencil, smax, grid)
end

# Generic stencil convolution.
#
# Two-layer design:
#   - `_convolve!` (non-generated outer): handles the parallel destination
#     loop via OhMyThreads.tforeach. Each `m` writes to its own `e[m]`,
#     so there's no race.
#   - `_convolve_at!` (@generated inner): per-axis unrolled body for a
#     single destination index `m`. Specialised on `(Per, Sz)`:
#
#     • periodic axis α:  loop `off ∈ -smax_α : smax_α`,
#                         source index `n_α = mod(m_α-1+off, Sz_α) + 1`
#                         (`Sz_α` splices in as an integer literal, so
#                          `mod` lowers to multiply-high; power-of-2 sizes
#                          use a bitmask).
#
#     • open axis α:      loop `off ∈ max(-smax_α, -(m_α-1)) : min(smax_α, Sz_α-m_α)`,
#                         source index `n_α = m_α + off`
#                         (no bounds check needed — the loop bounds
#                          guarantee the access is in-range).
#
# Splitting the closure out of the @generated body avoids the
# "@generated function body is not pure" error.
function _convolve!(e::AbstractArray{T,D},
                    q::AbstractArray{T,D},
                    stencil::AbstractArray{T,D},
                    smax::NTuple{D,Int},
                    grid::UniformGrid{D,T,Per,Sz}) where {D,T<:AbstractFloat,Per,Sz}
    @assert size(stencil) == ntuple(α -> 2 * smax[α] + 1, Val(D))
    fill!(e, zero(T))
    tforeach(CartesianIndices(e)) do m
        _convolve_at!(e, q, stencil, smax, grid, m)
    end
    return e
end

@generated function _convolve_at!(e::AbstractArray{T,D},
                                   q::AbstractArray{T,D},
                                   stencil::AbstractArray{T,D},
                                   smax::NTuple{D,Int},
                                   ::UniformGrid{D,T,Per,Sz},
                                   m::CartesianIndex{D}) where {D,T<:AbstractFloat,Per,Sz}
    @assert Per isa NTuple{D,Bool}
    @assert Sz  isa NTuple{D,Int}

    I_syms = ntuple(α -> Symbol("I_",  α), D)
    n_syms = ntuple(α -> Symbol("nx_", α), D)
    m_syms = ntuple(α -> Symbol("m_",  α), D)

    # Innermost statement: the actual stencil contribution.
    body = :(acc += stencil[$(I_syms...)] * q[$(n_syms...)])

    # Wrap axis loops from α=1 inward to α=D, so axis 1 (column-major
    # fastest-varying) ends up as the innermost loop.
    for α in 1:D
        off_sym = Symbol("off_", α)
        I_sym   = I_syms[α]
        n_sym   = n_syms[α]
        m_sym   = m_syms[α]
        s_α     = :(smax[$α])
        Sz_α    = Sz[α]       # splice as integer literal
        if Per[α]::Bool
            # Power-of-2 axes: skip mod entirely. Valid because the inner
            # offset `m - 1 + off` is ≥ -smax_α and Sz_α ≥ smax_α (the grid
            # is at least as wide as the stencil radius), so adding Sz_α
            # makes it non-negative without a branch.
            wrap_expr = ispow2(Sz_α) ?
                :((((($m_sym - 1 + $off_sym) + $Sz_α) & $(Sz_α - 1)) + 1)) :
                :(mod($m_sym - 1 + $off_sym, $Sz_α) + 1)
            body = quote
                for $off_sym in -$s_α:$s_α
                    $I_sym = $off_sym + $s_α + 1
                    $n_sym = $wrap_expr
                    $body
                end
            end
        else
            body = quote
                let lo = max(-$s_α, -($m_sym - 1)),
                    hi = min($s_α, $Sz_α - $m_sym)
                    for $off_sym in lo:hi
                        $I_sym = $off_sym + $s_α + 1
                        $n_sym = $m_sym + $off_sym
                        $body
                    end
                end
            end
        end
    end

    # Per-grid-point binding of m_α from the CartesianIndex.
    m_decls = Expr[:($(m_syms[α]) = m[$α]) for α in 1:D]

    return quote
        @inbounds begin
            $(m_decls...)
            acc = zero($T)
            $body
            e[m] = acc
        end
        return nothing
    end
end
