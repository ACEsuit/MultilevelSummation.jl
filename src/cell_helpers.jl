# Small orthorhombic-cell + periodic-image helpers shared by the
# top-level `core.jl` (short-range MSM force loop) and the
# `Reference.naive_*` direct sums. Lives outside Reference so it can be
# included before `core.jl` in the module load order.

function _assert_orthorhombic(cell::SMatrix{D,D,T}) where {D,T}
    @inbounds for α in 1:D, β in 1:D
        if α != β && !iszero(cell[α, β])
            throw(ArgumentError(
                "operation only supports orthorhombic cells; got cell[$α,$β] = $(cell[α, β])"
            ))
        end
    end
    return nothing
end

function _image_ranges(cell::SMatrix{D,D,T}, periodic::NTuple{D,Bool},
                       R_cut::T) where {D,T}
    return ntuple(D) do α
        if periodic[α]
            L = cell[α, α]
            L > 0 || throw(ArgumentError("periodic axis $α has L = $L ≤ 0"))
            isfinite(R_cut) || throw(ArgumentError(
                "R_cut must be finite for periodic axes (got $R_cut)"
            ))
            n_max = ceil(Int, R_cut / L)
            -n_max:n_max
        else
            0:0
        end
    end
end

@inline function _shift(cell::SMatrix{D,D,T}, n::NTuple{D,Int}) where {D,T}
    # Orthorhombic only: shift = Σ_α n_α · L_α e_α
    return SVector{D,T}(ntuple(α -> T(n[α]) * cell[α, α], D)...)
end
