using Test
using MultilevelSummation
using StaticArrays
using StableRNGs

# Convenience: build a periodic grid with the same h, n along every axis.
function _periodic_grid(::Val{D}, h::T, n::Int) where {D,T<:AbstractFloat}
    UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D), ntuple(_ -> zero(T), D),
                ntuple(_ -> true, D))
end

# Random positions inside the periodic box.
function _random_positions(rng, ::Val{D}, N::Int, L::T) where {D,T}
    [SVector{D,T}((rand(rng, D) .* L)...) for _ in 1:N]
end

@testset "anterp / interp: D=$D" for D in (1, 2, 3)
    rng = StableRNG(0xA17E5 + D)
    h   = 0.5
    n   = 8                        # box length = n*h = 4.0
    L   = h * n
    g   = _periodic_grid(Val(D), h, n)
    b   = CubicC1()

    @testset "transpose property" begin
        N = 12
        positions = _random_positions(rng, Val(D), N, L)
        # left: anterp x → grid, dot with random grid field y
        x = randn(rng, N)
        gv = grid_zeros(Float64, g)
        anterpolate!(gv, positions, x, g, b)
        y_grid = randn(rng, g.size...)
        lhs = sum(y_grid .* gv)
        # right: interp y_grid → particles, dot with x
        pots = zeros(Float64, N)
        interpolate!(pots, positions, y_grid, g, b)
        rhs = sum(pots .* x)
        @test isapprox(lhs, rhs; atol=1e-12, rtol=1e-12)
    end

    @testset "charge conservation (periodic)" begin
        N = 7
        positions = _random_positions(rng, Val(D), N, L)
        charges   = randn(rng, N)
        gv = grid_zeros(Float64, g)
        anterpolate!(gv, positions, charges, g, b)
        @test isapprox(sum(gv), sum(charges); atol=1e-10, rtol=1e-10)
    end

    @testset "constant field at particles (partition of unity)" begin
        # Set all grid values to a constant c. Then interp at any particle
        # should return c (partition of unity).
        N = 6
        positions = _random_positions(rng, Val(D), N, L)
        c = 2.7
        gv = fill(c, g.size...)
        pots = zeros(Float64, N)
        interpolate!(pots, positions, gv, g, b)
        for p in pots
            @test isapprox(p, c; atol=1e-12, rtol=1e-12)
        end
    end

    @testset "polynomial reproduction at particle sites" begin
        # Set grid values to p(x_m) for a polynomial p with degree ≤ 2 per axis.
        # Interpolation at any particle r_i should give p(r_i). Use an OPEN
        # grid so the periodic image of p doesn't break the test.
        N = 6
        # Per-axis quadratic polynomial p(r) = a₀ + Σ_α (b_α r_α + c_α r_α²).
        a0  = rand(rng) * 2 - 1
        bs  = rand(rng, D) .* 2 .- 1
        cs  = rand(rng, D) .* 2 .- 1
        eval_p = r -> begin
            v = a0
            @inbounds for α in 1:D
                v += bs[α] * r[α] + cs[α] * r[α]^2
            end
            v
        end
        # Place particles in the interior strip (away from boundary).
        positions = [SVector{D,Float64}((1.0 .+ rand(rng, D) .* (L - 2.0))...) for _ in 1:N]

        g_open = UniformGrid(ntuple(_ -> h, D), ntuple(_ -> n, D),
                             ntuple(_ -> 0.0, D), ntuple(_ -> false, D))
        gv = zeros(Float64, g_open.size...)
        @inbounds for I in CartesianIndices(gv)
            r_m = SVector{D,Float64}(ntuple(α -> (I[α] - 1) * h, D))
            gv[I] = eval_p(r_m)
        end
        pots = zeros(Float64, N)
        interpolate!(pots, positions, gv, g_open, b)
        for i in 1:N
            @test isapprox(pots[i], eval_p(positions[i]); atol=1e-10, rtol=1e-10)
        end
    end

    @testset "interpolate_grad: FD check" begin
        N = 5
        positions = _random_positions(rng, Val(D), N, L)
        gv = randn(rng, g.size...)
        grads = zeros(SVector{D,Float64}, N)
        interpolate_grad!(grads, positions, gv, g, b)
        pots = zeros(Float64, N)
        interpolate!(pots, positions, gv, g, b)
        for i in 1:N, α in 1:D
            δ = 1e-6
            e = SVector{D,Float64}(ntuple(β -> β == α ? δ : 0.0, D)...)
            p_plus  = copy(positions); p_plus[i]  = positions[i] + e
            p_minus = copy(positions); p_minus[i] = positions[i] - e
            pp = zeros(Float64, N); interpolate!(pp, p_plus,  gv, g, b)
            pm = zeros(Float64, N); interpolate!(pm, p_minus, gv, g, b)
            fd = (pp[i] - pm[i]) / (2δ)
            @test isapprox(grads[i][α], fd; atol=1e-6, rtol=1e-6)
        end
    end
end
