# tuning/tune_NaCl_perturbed.jl
#
# Variant of `tune_NaCl.jl`: NaCl supercells with Gaussian-perturbed
# positions corresponding very roughly to a 300 K thermal sample
# (σ ≈ 0.1 Å per Cartesian; close to the Debye-Waller estimate for NaCl
# near its Debye temperature). Charges unchanged so the system stays
# neutral.
#
# Why: on the perfect lattice (`tune_NaCl.jl`), the relative error is
# exactly N-invariant because both U and the absolute error scale with
# N. The plateau structure of `rel_err` we saw in the unperturbed sweep
# (clusters at 5.6e-5, 9.8e-5, etc.) is also a lattice-symmetry
# artefact. With independent random displacements those cancellations
# break and we should see (a) clean h- and a-dependence matching the
# paper's O(h^p / a^{p+1}) prediction and (b) genuine N-dependence of
# the relative error.
#
# Output: stdout table + `tune_NaCl_perturbed_results.csv`. Same CSV
# columns as the unperturbed sweep so `analyze_NaCl.jl` can read it.
#
# Run with:
#
#     julia --project=tuning tuning/tune_NaCl_perturbed.jl

using MultilevelSummation
using MultilevelSummation.Tune: sweep, ewald_reference, SweepResult
using StaticArrays
using Printf
using Random

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
const A_LAT       = 4.0           # NaCl-like lattice constant (Å)
const SIGMA       = 0.1           # Gaussian displacement σ per Cartesian (Å), ~300 K
const RNG_SEED    = 42

# Drop n_super = 8 here vs the unperturbed script: that single supercell
# accounted for ~3/4 of the unperturbed sweep wall-clock. (3,4,5,6)
# spans an 8× range in N — enough to see N-dependence of rel_err emerge.
const N_SUPER_LIST = (3, 4, 5, 6)

const H_VALUES = (0.5, 1.0, 2.0)
const A_VALUES = (2.0, 4.0, 8.0)

# ----------------------------------------------------------------------
# System construction
# ----------------------------------------------------------------------

"""
    build_nacl_perturbed(n_super; a_lat = A_LAT, σ = SIGMA, rng = ...)

Rock-salt NaCl supercell with each ion displaced by an independent
3-D Gaussian of standard deviation `σ` (Å) per Cartesian component.
Charges are ±1 and unchanged. Cell and periodicity are unchanged.
"""
function build_nacl_perturbed(n_super::Int;
                              a_lat::Float64 = A_LAT,
                              σ::Float64     = SIGMA,
                              rng::AbstractRNG = MersenneTwister(RNG_SEED))
    na_frac = (SVector(0.0, 0.0, 0.0), SVector(0.5, 0.5, 0.0),
               SVector(0.5, 0.0, 0.5), SVector(0.0, 0.5, 0.5))
    cl_frac = (SVector(0.5, 0.0, 0.0), SVector(0.0, 0.5, 0.0),
               SVector(0.0, 0.0, 0.5), SVector(0.5, 0.5, 0.5))
    positions = SVector{3,Float64}[]
    charges   = Float64[]
    @inbounds for i in 0:n_super-1, j in 0:n_super-1, k in 0:n_super-1
        offset = SVector(Float64(i), Float64(j), Float64(k)) .* a_lat
        for p in na_frac
            r0 = p .* a_lat .+ offset
            push!(positions, r0 .+ σ .* SVector(randn(rng), randn(rng), randn(rng)))
            push!(charges, +1.0)
        end
        for p in cl_frac
            r0 = p .* a_lat .+ offset
            push!(positions, r0 .+ σ .* SVector(randn(rng), randn(rng), randn(rng)))
            push!(charges, -1.0)
        end
    end
    box = n_super * a_lat
    cell = SMatrix{3,3,Float64}(box * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    return positions, charges, cell, periodic
end

# ----------------------------------------------------------------------
# Sweep (mirrors run_sweep in tune_NaCl.jl)
# ----------------------------------------------------------------------
function run_sweep()
    rows = NamedTuple[]
    for n_super in N_SUPER_LIST
        positions, charges, cell, periodic = build_nacl_perturbed(n_super)
        N   = length(positions)
        box = cell[1, 1]
        @printf "\n=========  n_super = %d, N = %d ions, box = %.2f Å, σ = %.2f Å, seed = %d  =========\n" n_super N box SIGMA RNG_SEED

        t_ewald = @elapsed U_ref = ewald_reference(positions, charges, cell; tol = 1e-9)
        @printf "Ewald reference: U_ref = %.6e (run in %.2fs)\n\n" U_ref t_ewald

        results::Vector{SweepResult{Float64}} =
            sweep(positions, charges, cell, periodic;
                  reference = U_ref,
                  h_values  = H_VALUES,
                  a_values  = A_VALUES,
                  L_strategy = :all)

        @printf "%4s %6s %3s %5s   %12s %10s\n" "h" "a" "L" "n_grid" "rel_err" "t_msm (s)"
        @printf "%4s %6s %3s %5s   %12s %10s\n" "----" "------" "---" "------" "------------" "----------"
        for r in results
            @printf "%4.2f %6.2f %3d %5d   %12.3e %10.4f\n" r.h r.a r.L r.n_grid r.rel_err r.t_msm
            push!(rows, (N = N, n_super = n_super, box = box,
                         h = r.h, n_grid = r.n_grid, a = r.a, L = r.L,
                         rel_err = r.rel_err, t_msm = r.t_msm,
                         t_ewald = t_ewald, U_ref = U_ref))
        end
    end
    return rows
end

# ----------------------------------------------------------------------
# Run & dump CSV
# ----------------------------------------------------------------------
const rows = run_sweep()

csv_path = joinpath(@__DIR__, "tune_NaCl_perturbed_results.csv")
open(csv_path, "w") do io
    println(io, "N,n_super,box,h,n_grid,a,L,rel_err,t_msm,t_ewald,U_ref")
    for r in rows
        @printf io "%d,%d,%.4f,%.4f,%d,%.4f,%d,%.6e,%.6e,%.6e,%.6e\n" r.N r.n_super r.box r.h r.n_grid r.a r.L r.rel_err r.t_msm r.t_ewald r.U_ref
    end
end
@info "wrote $(length(rows)) rows to $csv_path"
