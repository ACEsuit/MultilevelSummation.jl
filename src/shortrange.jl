# Short-range direct pair sum + the long-range force gather. Extracted from
# core.jl so the corresponding KA-backed variant in `shortrange_ka.jl` can
# sit alongside without core.jl gaining a second responsibility.

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
                                      calc::MSMCalculator{T};
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
