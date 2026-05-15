using Test
using MultilevelSummation
using MultilevelSummation: _short_range_pair_ka
using KernelAbstractions
using StaticArrays
using StableRNGs

# Independent O(N²) reference: explicit double loop + image enumeration.
# Mirrors `_short_range_and_long_forces` from src/shortrange.jl exactly,
# without the long-range tail — keeps the comparison self-contained.
function _ref_short_range(positions::AbstractVector{SVector{D,T}},
                          charges::AbstractVector{T},
                          cell::SMatrix{D,D,T},
                          periodic::NTuple{D,Bool},
                          splitting,
                          a::T) where {D, T}
    N = length(positions)
    image_ranges = MultilevelSummation._image_ranges(cell, periodic, a)
    a² = a * a

    U_short = zero(T)
    F_short = zeros(SVector{D,T}, N)
    @inbounds for i in 1:N, j in 1:N
        qiqj = charges[i] * charges[j]
        for n in Iterators.product(image_ranges...)
            (i == j && all(==(0), n)) && continue
            shift = MultilevelSummation._shift(cell, n)
            r     = positions[i] - positions[j] + shift
            sum(abs2, r) > a² && continue
            U_short += qiqj * short_range(splitting, r)
            F_short[i] -= qiqj * short_range_grad(splitting, r)
        end
    end
    U_short *= T(1//2)
    return U_short, F_short
end

@testset "short-range (KA via NeighbourLists): D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xCA50A3 + D)
    L_box = 4.0
    a = 1.2
    splitting = HardyC2Cubic(a, 3)

    for periodic_tuple in (ntuple(_ -> true, D), ntuple(_ -> false, D))
        cell     = SMatrix{D,D,Float64}(L_box * one(SMatrix{D,D,Float64}))
        N        = 12
        positions = [SVector{D,Float64}((rand(rng, D) .* L_box)...) for _ in 1:N]
        charges   = randn(rng, N) ./ sqrt(N)

        @testset "matches reference, periodic=$periodic_tuple" begin
            U_ref, F_ref = _ref_short_range(positions, charges, cell,
                                            periodic_tuple, splitting, a)
            U_ka, F_ka = _short_range_pair_ka(positions, charges, cell,
                                              periodic_tuple, splitting, a, CPU())
            @test isapprox(U_ka, U_ref; atol=1e-12, rtol=1e-10)
            @test isapprox(F_ka, F_ref; atol=1e-12, rtol=1e-10)
        end
    end

    # Mixed BC: periodic on axis 1 only (D >= 2), exercises the
    # NL.jl pbc handling on a non-uniform pbc tuple.
    if D >= 2
        mixed_pbc = ntuple(α -> α == 1, D)
        cell      = SMatrix{D,D,Float64}(L_box * one(SMatrix{D,D,Float64}))
        N         = 12
        positions = [SVector{D,Float64}((rand(rng, D) .* L_box)...) for _ in 1:N]
        charges   = randn(rng, N) ./ sqrt(N)

        @testset "matches reference, mixed BC" begin
            U_ref, F_ref = _ref_short_range(positions, charges, cell,
                                            mixed_pbc, splitting, a)
            U_ka, F_ka = _short_range_pair_ka(positions, charges, cell,
                                              mixed_pbc, splitting, a, CPU())
            @test isapprox(U_ka, U_ref; atol=1e-12, rtol=1e-10)
            @test isapprox(F_ka, F_ref; atol=1e-12, rtol=1e-10)
        end
    end
end
