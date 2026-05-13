using Test
using MultilevelSummation
using StaticArrays
using StableRNGs

# Independent O(N²) reference: e[m] = Σ_n k_l(r_m − r_n) q[n] with the
# splitting evaluated on demand. Different control flow from grid_cutoff!.
function _ref_grid_cutoff(q::AbstractArray{T,D},
                          grid::UniformGrid{D,T},
                          splitting, l::Int) where {D,T}
    e = zero(q)
    @inbounds for m in CartesianIndices(e)
        r_m = grid.origin .+ SVector{D,T}(ntuple(α -> T(m[α] - 1) * grid.spacing[α], D))
        acc = zero(T)
        for n in CartesianIndices(q)
            r_n = grid.origin .+ SVector{D,T}(ntuple(α -> T(n[α] - 1) * grid.spacing[α], D))
            Δr = r_m - r_n
            # Periodic minimum image for periodic axes
            Δr = _minimum_image(Δr, grid)
            # The level-l kernel has compact support |r| ≤ 2^l · a
            a_lp1 = T(2)^l * splitting.a
            sum(abs2, Δr) ≤ a_lp1^2 || continue
            acc += long_range_level(splitting, l, Δr) * q[n]
        end
        e[m] = acc
    end
    return e
end

function _minimum_image(Δr::SVector{D,T}, grid::UniformGrid{D,T}) where {D,T}
    Δr_out = Δr
    for α in 1:D
        if grid.periodic[α]
            L = grid.spacing[α] * grid.size[α]
            Δr_out = setindex(Δr_out, Δr_out[α] - L * round(Δr_out[α] / L), α)
        end
    end
    return Δr_out
end

@testset "grid cutoff: D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xC0F61 + D)
    h, n = 0.5, 12                              # n large enough vs stencil
    g = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                    ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
    a = h                                       # smax = ceil(2 a/h) = 2 → stencil 5
    L = 3
    splitting = HardyC2Cubic(a, L)

    @testset "single charge → stencil shape" begin
        for l in 1:(L-1)
            q = zeros(g.size...)
            mid = ntuple(_ -> n ÷ 2 + 1, D)
            q[mid...] = 1.0
            e = zeros(g.size...)
            grid_cutoff!(e, q, g, splitting, l)
            stencil, smax = build_stencil(splitting, l, g.spacing)
            # The output at each grid point m equals stencil[m - mid_0] (with periodic wrap).
            @inbounds for I in CartesianIndices(stencil)
                off = ntuple(α -> I[α] - 1 - smax[α], D)
                m_raw = ntuple(α -> mid[α] - 1 + off[α], D)
                m_wrapped, ok = wrap_index(m_raw, g)
                ok || continue
                @test isapprox(e[m_wrapped...], stencil[I]; atol=1e-12, rtol=1e-12)
            end
        end
    end

    @testset "linearity" begin
        q1 = randn(rng, g.size...)
        q2 = randn(rng, g.size...)
        α, β = 0.7, -1.3
        e_sum = zeros(g.size...); grid_cutoff!(e_sum, α .* q1 .+ β .* q2, g, splitting, 1)
        e1 = zeros(g.size...);    grid_cutoff!(e1, q1, g, splitting, 1)
        e2 = zeros(g.size...);    grid_cutoff!(e2, q2, g, splitting, 1)
        @test isapprox(e_sum, α .* e1 .+ β .* e2; atol=1e-10, rtol=1e-10)
    end

    @testset "agrees with independent O(N²) reference, level=$l" for l in 1:(L-1)
        q   = randn(rng, g.size...)
        e   = zeros(g.size...);  grid_cutoff!(e, q, g, splitting, l)
        ref = _ref_grid_cutoff(q, g, splitting, l)
        @test isapprox(e, ref; atol=1e-10, rtol=1e-10)
    end

    @testset "open BC: out-of-bounds neighbours dropped" begin
        g_open = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                             ntuple(_ -> 0.0, D), ntuple(_ -> false, D))
        q = zeros(g_open.size...)
        # Put a charge in the corner: many neighbours are OOB
        q[ntuple(_ -> 1, D)...] = 1.0
        e = zeros(g_open.size...)
        grid_cutoff!(e, q, g_open, splitting, 1)
        # Compare against an independent reference run on the open grid.
        ref = _ref_grid_cutoff(q, g_open, splitting, 1)
        @test isapprox(e, ref; atol=1e-12, rtol=1e-12)
    end
end
