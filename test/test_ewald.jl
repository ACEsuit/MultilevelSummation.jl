using Test
using StaticArrays
using StableRNGs

include("refs/ewald.jl")
using .EwaldRef: ewald_energy, ewald_energy_forces

# Build a random neutral system in an orthorhombic cell.
function _random_neutral_system(rng, N::Int, L::Float64)
    positions = [SVector{3,Float64}((rand(rng, 3) .* L)...) for _ in 1:N]
    charges   = randn(rng, N)
    charges  .-= sum(charges) / N
    cell      = SMatrix{3,3,Float64}([L 0 0; 0 L 0; 0 0 L])
    return positions, charges, cell
end

@testset "Ewald reference" begin
    rng = StableRNG(0xE9A1D)

    @testset "α-invariance (defining property of Ewald)" begin
        N = 8
        L = 4.0
        positions, charges, cell = _random_neutral_system(rng, N, L)

        # Sweep α with R_cut, k_cut scaled so each piece is well-converged.
        αs = (0.4, 0.6, 0.8, 1.0)
        Us = Float64[]
        for α in αs
            R_cut = 10 / α
            k_cut = 12 * α
            U = ewald_energy(positions, charges, cell; α=α, R_cut=R_cut, k_cut=k_cut)
            push!(Us, U)
        end
        # Pairwise differences should be tiny.
        for a in 1:length(αs), b in (a+1):length(αs)
            @test isapprox(Us[a], Us[b]; atol=1e-8, rtol=1e-8)
        end
    end

    @testset "FD gradient check on forces" begin
        N = 5
        L = 3.0
        positions, charges, cell = _random_neutral_system(rng, N, L)
        α     = 0.7
        R_cut = 12.0
        k_cut = 10.0

        U, F = ewald_energy_forces(positions, charges, cell; α=α, R_cut=R_cut, k_cut=k_cut)
        f_of(p) = ewald_energy(p, charges, cell; α=α, R_cut=R_cut, k_cut=k_cut)

        for i in 1:N, d in 1:3
            δ = 1e-5
            e = SVector{3,Float64}(ntuple(k -> k == d ? δ : 0.0, 3)...)
            p_plus  = copy(positions); p_plus[i]  = positions[i] + e
            p_minus = copy(positions); p_minus[i] = positions[i] - e
            fd = (f_of(p_plus) - f_of(p_minus)) / (2δ)
            @test isapprox(-F[i][d], fd; atol=1e-5, rtol=1e-5)
        end
    end

    @testset "rejects non-neutral charges" begin
        N = 3
        L = 2.0
        positions = [SVector{3,Float64}((rand(rng, 3) .* L)...) for _ in 1:N]
        charges   = ones(Float64, N)              # non-neutral
        cell      = SMatrix{3,3,Float64}([L 0 0; 0 L 0; 0 0 L])
        @test_throws ArgumentError ewald_energy(
            positions, charges, cell; α=0.5, R_cut=5.0, k_cut=8.0,
        )
    end

    @testset "rejects non-orthorhombic cell" begin
        positions = [SVector(0.0, 0.0, 0.0), SVector(0.5, 0.5, 0.5)]
        charges   = [1.0, -1.0]
        cell      = SMatrix{3,3,Float64}([1.0 0.1 0; 0 1 0; 0 0 1])
        @test_throws ArgumentError ewald_energy(
            positions, charges, cell; α=0.5, R_cut=5.0, k_cut=8.0,
        )
    end
end
