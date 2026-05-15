# End-to-end KA orchestration. Mirrors `_msm_compute_cpu` from core.jl
# but allocates each per-level buffer on the KA backend, chains the
# `*_ka.jl` operators, and uses the cell-list-based short-range pair
# sum via NeighbourLists.jl.

using KernelAbstractions: KernelAbstractions, Backend

function _msm_compute_ka(positions::AbstractVector{SVector{D,T}},
                         charges::AbstractVector{T},
                         cell::SMatrix{D,D,T},
                         periodic::NTuple{D,Bool},
                         calc::MSMCalculator{T};
                         want_forces::Bool,
                         backend::Backend) where {D, T<:AbstractFloat}
    @assert length(positions) == length(charges)
    N = length(positions)
    splitting = calc.splitting
    basis     = calc.basis
    L         = nlevels(calc)

    # --- Hierarchy + KA-backed buffers ------------------------------------
    grids = build_grid_hierarchy(cell, periodic, positions, calc)
    qs = [KernelAbstractions.zeros(backend, T, g.size...) for g in grids]
    es = [KernelAbstractions.zeros(backend, T, g.size...) for g in grids]

    # 1. Anterpolate particles to the finest grid
    anterpolate_ka!(qs[1], positions, charges, grids[1], basis, backend)

    # 2. Restrict q^1 → q^2 → ... → q^L
    for l in 1:(L - 1)
        restrict_ka!(qs[l+1], qs[l], grids[l+1], grids[l], basis, backend)
    end

    # 3. Neutralising background at the top, if applicable
    if requires_neutralising_background(splitting) && all(periodic)
        apply_neutralising_background_ka!(qs[end], splitting, grids[end], backend)
    end

    # 4. Grid-cutoff convolutions for l = 1, …, L-1
    for l in 1:(L - 1)
        grid_cutoff_ka!(es[l], qs[l], grids[l], splitting, l, backend)
    end

    # 5. Top-level direct sum
    top_level_ka!(es[end], qs[end], grids[end], splitting, backend)

    # 6. Prolong potentials downward: e^l += prolong(e^{l+1})
    for l in (L - 1):-1:1
        tmp = KernelAbstractions.zeros(backend, T, grids[l].size...)
        prolong_ka!(tmp, es[l+1], grids[l], grids[l+1], basis, backend)
        es[l] .+= tmp
    end

    # 7. Interpolate e^1 to particle potentials
    pots = KernelAbstractions.zeros(backend, T, N)
    interpolate_ka!(pots, positions, es[1], grids[1], basis, backend)

    # 8. Long-range energy (host-side scalar reduction over a backend array)
    U_long = T(1//2) * sum(charges .* pots)

    # 9. Self-interaction correction
    k0_self = kernel_self_value(splitting, Val(D), T)
    U_self  = -T(1//2) * k0_self * sum(abs2, charges)

    # 10. Short-range pair sum + long-range force
    a = short_range_cutoff(calc)
    U_short, F_short, F_long = _short_range_and_long_forces_ka(
        positions, charges, cell, periodic, splitting, basis, grids, es, calc;
        want_forces, a, backend,
    )

    U_total = U_short + U_long + U_self

    if want_forces
        return U_total, F_short .+ F_long
    else
        return U_total, similar(charges, SVector{D,T}, 0)
    end
end
