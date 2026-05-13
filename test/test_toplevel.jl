using Test
using MLSum
using StaticArrays
using StableRNGs

# Independent reference for the top-level direct sum.
function _ref_top_level(q::AbstractArray{T,D},
                        grid::UniformGrid{D,T},
                        splitting) where {D,T}
    e = zero(q)
    @inbounds for m in CartesianIndices(e)
        r_m = grid.origin .+ SVector{D,T}(ntuple(α -> T(m[α] - 1) * grid.spacing[α], D))
        acc = zero(T)
        for n in CartesianIndices(q)
            r_n = grid.origin .+ SVector{D,T}(ntuple(α -> T(n[α] - 1) * grid.spacing[α], D))
            Δr = r_m - r_n
            for α in 1:D
                if grid.periodic[α]
                    Lα = grid.spacing[α] * grid.size[α]
                    Δr = setindex(Δr, Δr[α] - Lα * round(Δr[α] / Lα), α)
                end
            end
            acc += top_level(splitting, Δr) * q[n]
        end
        e[m] = acc
    end
    return e
end

@testset "top level: D=$D" for D in (1, 2, 3)
    rng = StableRNG(0x70F + D)
    h, n = 0.5, 6
    a = h
    L = 2
    splitting = HardyC2Cubic(a, L)

    @testset "open BC: matches independent reference" begin
        g = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                        ntuple(_ -> 0.0, D), ntuple(_ -> false, D))
        q = randn(rng, g.size...)
        e = zeros(g.size...)
        top_level!(e, q, g, splitting)
        ref = _ref_top_level(q, g, splitting)
        @test isapprox(e, ref; atol=1e-10, rtol=1e-10)
    end

    @testset "fully periodic + neutralising bg ⇒ e == 0" begin
        # The "top grid" for fully periodic systems is 1 point per axis.
        g_top = UniformGrid(ntuple(_ -> h * n, D), ntuple(_ -> 1, D),
                            ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
        q = fill(0.37, g_top.size...)             # arbitrary nonzero
        apply_neutralising_background!(q, splitting, g_top)
        @test all(iszero, q)
        e = zeros(g_top.size...)
        top_level!(e, q, g_top, splitting)
        @test all(iszero, e)
    end

    @testset "linearity" begin
        g = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                        ntuple(_ -> 0.0, D), ntuple(_ -> false, D))
        q1 = randn(rng, g.size...)
        q2 = randn(rng, g.size...)
        α, β = 0.7, -1.3
        e_sum = zeros(g.size...);  top_level!(e_sum, α .* q1 .+ β .* q2, g, splitting)
        e1    = zeros(g.size...);  top_level!(e1, q1, g, splitting)
        e2    = zeros(g.size...);  top_level!(e2, q2, g, splitting)
        @test isapprox(e_sum, α .* e1 .+ β .* e2; atol=1e-10, rtol=1e-10)
    end

    @testset "top_grid_size halves correctly" begin
        g_fine = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> 16, D),
                             ntuple(_ -> 0.0, D), ntuple(_ -> true, D))
        @test top_grid_size(g_fine, 1) == ntuple(_ -> 16, D)
        @test top_grid_size(g_fine, 2) == ntuple(_ -> 8, D)
        @test top_grid_size(g_fine, 3) == ntuple(_ -> 4, D)
        @test top_grid_size(g_fine, 5) == ntuple(_ -> 1, D)
    end

    @testset "apply_neutralising_background! no-op for open BC" begin
        g_open = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> 4, D),
                             ntuple(_ -> 0.0, D), ntuple(_ -> false, D))
        q = fill(0.5, g_open.size...)
        apply_neutralising_background!(q, splitting, g_open)
        @test all(==(0.5), q)
    end
end
