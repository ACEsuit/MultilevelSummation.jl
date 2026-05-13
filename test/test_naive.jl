using Test
using MultilevelSummation
using StaticArrays
using StableRNGs

# Independent reference: same math, different loop structure.
# This implementation iterates over ordered pairs (i, j) with i < j (no double
# counting, no /2). Used to cross-check MultilevelSummation.naive_energy / naive_energy_forces.
function _indep_ref_energy_open(positions::Vector{SVector{D,T}},
                                charges::Vector{T},
                                kernel) where {D,T}
    N = length(positions)
    U = zero(T)
    for i in 1:N, j in (i+1):N
        U += charges[i] * charges[j] * kernel(positions[i] - positions[j])
    end
    return U
end

function _indep_ref_forces_open(positions::Vector{SVector{D,T}},
                                charges::Vector{T},
                                kernel) where {D,T}
    N = length(positions)
    F = zeros(SVector{D,T}, N)
    for i in 1:N, j in 1:N
        i == j && continue
        F[i] -= charges[i] * charges[j] * MultilevelSummation.grad(kernel, positions[i] - positions[j])
    end
    return F
end

# FD-gradient of a scalar energy w.r.t. positions[i] along axis α.
function _fd_grad(f, positions::Vector{SVector{D,T}}, i, α; δ=1e-6) where {D,T}
    e = SVector{D,T}(ntuple(β -> β == α ? T(δ) : zero(T), D)...)
    pos_plus  = copy(positions); pos_plus[i]  = positions[i] + e
    pos_minus = copy(positions); pos_minus[i] = positions[i] - e
    return (f(pos_plus) - f(pos_minus)) / (2δ)
end

@testset "naive reference: open BC, scalar charges" begin
    rng = StableRNG(0xBEEF)

    @testset "agrees with independent reference, D=$D, kernel=$(typeof(K).name.name)" for
            D in (1, 2, 3),
            K in (Coulomb(), InversePower{6,Float64}(), RationalDecay{4}(1.2))
        N = 5
        positions = [SVector{D,Float64}(randn(rng, D)...) for _ in 1:N]
        # ensure no coincident points
        charges   = randn(rng, N) ./ sqrt(N)
        cell      = zero(SMatrix{D,D,Float64})        # zero — unused for open
        periodic  = ntuple(_ -> false, D)

        U_mlsum, F_mlsum = naive_energy_forces(positions, charges, cell, periodic, K)
        U_ref            = _indep_ref_energy_open(positions, charges, K)
        F_ref            = _indep_ref_forces_open(positions, charges, K)

        @test isapprox(U_mlsum, U_ref; atol=1e-12, rtol=1e-12)
        for i in 1:N
            @test isapprox(F_mlsum[i], F_ref[i]; atol=1e-10, rtol=1e-10)
        end
    end

    @testset "FD gradient check, D=$D" for D in (1, 2, 3)
        K = Coulomb()
        N = 6
        positions = [SVector{D,Float64}(randn(rng, D)...) for _ in 1:N]
        charges   = randn(rng, N) ./ sqrt(N)
        cell      = zero(SMatrix{D,D,Float64})
        periodic  = ntuple(_ -> false, D)

        U, F = naive_energy_forces(positions, charges, cell, periodic, K)
        f_of(p) = naive_energy(p, charges, cell, periodic, K)

        for i in 1:N, α in 1:D
            fd = _fd_grad(f_of, positions, i, α; δ=1e-6)
            @test isapprox(-F[i][α], fd; atol=1e-5, rtol=1e-5)
        end
    end

    @testset "two-particle analytic match, D=$D" for D in (1, 2, 3)
        K = Coulomb()
        positions = [SVector{D,Float64}(zeros(D)...),
                     SVector{D,Float64}(ntuple(i -> i == 1 ? 1.7 : 0.0, D)...)]
        charges   = [1.0, -1.0]
        cell      = zero(SMatrix{D,D,Float64})
        periodic  = ntuple(_ -> false, D)
        U, F = naive_energy_forces(positions, charges, cell, periodic, K)
        @test U ≈ -1 / 1.7
        # +1 charge at origin, -1 charge at +x: they attract, so F[1] points +x.
        @test isapprox(F[1][1], +1 / 1.7^2; atol=1e-12, rtol=1e-12)
        @test isapprox(F[2][1], -1 / 1.7^2; atol=1e-12, rtol=1e-12)
    end
end

@testset "naive reference: fully periodic, fast-decay convergence" begin
    rng = StableRNG(0xFACE)

    # Use RationalDecay (smooth at origin, fast tail) so truncated-image
    # sums converge geometrically — exactly the regime where the naive
    # reference is reliable.
    @testset "image-sum convergence, D=$D, N=$N" for D in (2, 3), N in (3, 4)
        K = RationalDecay{6}(0.7)
        L = 3.0
        cell     = SMatrix{D,D,Float64}(L * one(SMatrix{D,D,Float64}))
        periodic = ntuple(_ -> true, D)
        positions = [SVector{D,Float64}((rand(rng, D) .* L)...) for _ in 1:N]
        charges   = randn(rng, N) ./ sqrt(N)
        charges  .-= sum(charges) / N             # neutralise

        Us = Float64[]
        for R_cut in (2.0, 4.0, 8.0, 16.0)
            push!(Us, naive_energy(positions, charges, cell, periodic, K; R_cut=R_cut))
        end
        # Cauchy-style: |U_k - U_{k+1}| should shrink as R_cut doubles.
        for k in 1:(length(Us)-1)
            @test abs(Us[k] - Us[end]) > abs(Us[k+1] - Us[end]) * 0.5
        end
        # Converged value: the largest R_cut.
        @test abs(Us[end-1] - Us[end]) < 1e-4 * max(abs(Us[end]), 1)
    end
end

@testset "naive reference: errors and edge cases" begin
    # non-orthorhombic should throw
    @test_throws ArgumentError naive_energy(
        [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)],
        [1.0, -1.0],
        SMatrix{3,3,Float64}([1 0.2 0; 0 1 0; 0 0 1]),
        (true, true, true),
        Coulomb();
        R_cut=2.0,
    )

    # periodic axis with R_cut=Inf must throw
    @test_throws ArgumentError naive_energy(
        [SVector(0.0, 0.0, 0.0), SVector(0.5, 0.5, 0.5)],
        [1.0, -1.0],
        one(SMatrix{3,3,Float64}),
        (true, true, true),
        Coulomb(),
    )
end
