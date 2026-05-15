# KernelAbstractions-backed short-range pair sum.
#
# Strategy: build a `SortedCellList` via NeighbourLists.jl and launch a
# custom @kernel with one workitem per atom. The inner loop runs
# `for_each_neighbour(clist, i)` — NL.jl's lazy, GPU-callable neighbour
# iterator — and accumulates U_short[i] and F_short[i] locally. Each
# workitem owns its output index exclusively, so no atomics are needed.
#
# Dimension handling: NL.jl is hard-coded to 3D (`SVector{3,T}`). For
# D ∈ {1, 2} we pad positions with zeros, pad the cell with an
# orthogonal "large enough" identity block on the unused diagonal, and
# set the padded pbc axes to `false`. R3 inside the kernel always has
# zeros in the padded slots, so dropping to `SVector{D,T}` gives the
# correct displacement for the splitting evaluation.

using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, Backend
using NeighbourLists: neighbour_list, for_each_neighbour, SortedCellList
using Adapt: Adapt

# --- Adapt rule for NL.jl's SortedCellList -------------------------------
#
# NL.jl ships no `Adapt.adapt_structure` for `SortedCellList`, which
# means CUDA / ROCm / Metal kernel launches can't recursively adapt the
# CuArray (etc.) fields to their device-side counterparts
# (`CuDeviceArray`, …). Without this, passing a `SortedCellList` as a
# kernel argument fails with "non-bitstype argument" at GPU compile
# time. The rule below makes the struct fully adaptable.
#
# Type piracy disclaimer: we extend `Adapt.adapt_structure` for an
# external type. Acceptable here because there is no upstream definition
# to conflict with and the semantics are unambiguous. Track the
# upstream PR / issue request in PRIORITIES.md.
@inline Adapt.adapt_structure(to, clist::SortedCellList) =
    SortedCellList(
        Adapt.adapt(to, clist.X),
        Adapt.adapt(to, clist.X_orig),
        Adapt.adapt(to, clist.perm),
        Adapt.adapt(to, clist.cell_id),
        Adapt.adapt(to, clist.cell_offsets),
        clist.cell,
        clist.inv_cell,
        clist.pbc,
        clist.cutoff,
        clist.ncells,
        clist.ncells_total,
    )

# --- 3D-padding adapters --------------------------------------------------

# Position padding. Returns the input unchanged for D=3; otherwise allocates
# a new `SVector{3,T}` vector on the same backend as `positions` and fills
# the padded slots with zero.
function _pad_positions_3d(positions::AbstractVector{SVector{D,T}}) where {D, T}
    D == 3 && return positions
    host_padded = Vector{SVector{3,T}}(undef, length(positions))
    @inbounds for i in eachindex(positions)
        p = positions[i]
        host_padded[i] = SVector{3,T}(ntuple(α -> α <= D ? p[α] : zero(T), Val(3)))
    end
    backend_padded = similar(positions, SVector{3,T}, length(positions))
    copyto!(backend_padded, host_padded)
    return backend_padded
end

# Cell padding. NL.jl wants a valid (non-degenerate) cell matrix to
# construct cell-list bins. Two padding jobs in one:
#
#   1. For D < 3, fill the unused 3D axes with `L_pad` on the diagonal so
#      `inv(cell)` and `det(cell)` stay well-behaved.
#
#   2. For any *open* axis (`periodic[α] == false`) whose original cell
#      extent is too small (or zero — the open-BC test uses
#      `cell = zero(SMatrix)`), substitute `L_pad`. Open axes are not
#      wrapped by NL.jl, so the substituted extent only affects bin
#      placement and not the pair geometry.
function _pad_cell_3d(cell::SMatrix{D,D,T},
                      periodic::NTuple{D,Bool},
                      a::T) where {D, T}
    L_pad = max(T(2) * a, one(T))
    vals = ntuple(Val(9)) do k
        i = (k - 1) % 3 + 1
        j = (k - 1) ÷ 3 + 1
        if i <= D && j <= D
            if i == j
                periodic[i] ? cell[i, j] : max(cell[i, j], L_pad)
            else
                cell[i, j]
            end
        elseif i == j
            L_pad
        else
            zero(T)
        end
    end
    return SMatrix{3,3,T}(vals...)
end

# Per-axis pbc padding. Padded axes are non-periodic.
@inline _pad_pbc_3d(periodic::NTuple{D,Bool}) where {D} =
    SVector{3,Bool}(ntuple(α -> α <= D ? periodic[α] : false, Val(3)))

# --- short-range kernel ---------------------------------------------------

@kernel function _short_range_ka_kernel!(U_per_atom,
                                          F_short,
                                          @Const(charges),
                                          clist,
                                          splitting,
                                          ::Val{D}) where {D}
    i = @index(Global)
    T = eltype(charges)
    qi = charges[i]

    # Refs are used (rather than plain Julia mutables) so the closure
    # captured by `for_each_neighbour`'s do-block can update them without
    # boxing — matches the pattern NL.jl's own GPU kernels use.
    u_i = Ref(zero(T))
    f_i = Ref(zero(SVector{D,T}))

    for_each_neighbour(clist, i) do j, R3, S
        # NL.jl returns R = r_j - r_i. The CPU pair loop uses
        # r = r_i - r_j; the gradient is anti-symmetric under r → -r,
        # so we negate here to match the existing sign convention.
        R_d = SVector{D,T}(ntuple(α -> -R3[α], Val(D)))
        qj  = charges[j]
        u_i[] += qi * qj * short_range(splitting, R_d)
        f_i[] -= qi * qj * short_range_grad(splitting, R_d)
    end

    @inbounds U_per_atom[i] = u_i[]
    @inbounds F_short[i]    = f_i[]
end

# --- public entry point ---------------------------------------------------

"""
    _short_range_pair_ka(positions, charges, cell, periodic, splitting, a, backend)
        -> (U_short::T, F_short::AbstractVector{SVector{D,T}})

Compute the short-range pair sum and per-atom short-range force via
`SortedCellList` + the custom `@kernel`. Pure pair-sum step — no
long-range gather, no AtomsBase / calculator coupling — so this is
the testable unit for PR-D.

`F_short[i] = -q_i Σ_{j ≠ i (or image)} q_j ∇K_0(r_ij)`, and
`U_short = ½ Σ_i Σ_j q_i q_j K_0(r_ij)`.
"""
function _short_range_pair_ka(positions::AbstractVector{SVector{D,T}},
                              charges::AbstractVector{T},
                              cell::SMatrix{D,D,T},
                              periodic::NTuple{D,Bool},
                              splitting,
                              a::T,
                              backend::Backend) where {D, T<:AbstractFloat}
    N = length(positions)
    X3    = _pad_positions_3d(positions)
    cell3 = _pad_cell_3d(cell, periodic, a)
    pbc3  = _pad_pbc_3d(periodic)
    clist = neighbour_list(X3, a, cell3, pbc3; lazy = true)

    U_per_atom = KernelAbstractions.zeros(backend, T, N)
    F_short    = KernelAbstractions.zeros(backend, SVector{D,T}, N)

    kernel = _short_range_ka_kernel!(backend)
    kernel(U_per_atom, F_short, charges, clist, splitting, Val(D);
           ndrange = N)
    _ka_synchronize(backend)

    # Each ordered (i,j) pair is visited twice in the per-atom sum.
    U_short = T(1//2) * sum(U_per_atom)
    return U_short, F_short
end

"""
    _short_range_and_long_forces_ka(positions, charges, cell, periodic,
                                    splitting, basis, grids, es, calc;
                                    want_forces, a, backend)
        -> (U_short, F_short, F_long)

KernelAbstractions counterpart of `_short_range_and_long_forces`. Built
on top of `_short_range_pair_ka` for the pair sum, with the long-range
force gather done by `interpolate_grad_ka!` on `es[1]`.
"""
function _short_range_and_long_forces_ka(positions::AbstractVector{SVector{D,T}},
                                         charges::AbstractVector{T},
                                         cell::SMatrix{D,D,T},
                                         periodic::NTuple{D,Bool},
                                         splitting,
                                         basis,
                                         grids,
                                         es,
                                         calc::MSMCalculator{T};
                                         want_forces::Bool,
                                         a::T,
                                         backend::Backend) where {D, T<:AbstractFloat}
    N = length(positions)
    U_short, F_short = _short_range_pair_ka(positions, charges, cell, periodic,
                                            splitting, a, backend)

    F_long = KernelAbstractions.zeros(backend, SVector{D,T}, N)
    if want_forces
        interp_grads = KernelAbstractions.zeros(backend, SVector{D,T}, N)
        interpolate_grad_ka!(interp_grads, positions, es[1], grids[1], basis, backend)
        F_long .= .-charges .* interp_grads
    end

    return U_short, F_short, F_long
end
