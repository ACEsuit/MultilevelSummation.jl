# tuning/tune_NaCl.jl
#
# Hyperparameter sweep on perfect rock-salt NaCl supercells of varying
# size, comparing MSM energy against a naive Ewald reference.
#
# Why NaCl: a tractable, fully-periodic, charge-neutral, strongly-bound
# ionic system. The Madelung energy is well-known and Ewald converges
# cleanly. Lets us study how the MSM hyperparameters (h, a, L) trade
# accuracy against cost as N grows.
#
# This script is a thin caller around `MultilevelSummation.Tune.sweep`
# and `MultilevelSummation.Tune.ewald_reference`. All the system-agnostic
# sweep mechanics live in the `Tune` submodule.
#
# Output:
#   - Per-system table on stdout.
#   - CSV `tune_NaCl_results.csv` next to this script for later plotting.
#
# Run with:
#
#     cd MultilevelSummation.jl
#     julia --project=tuning tuning/tune_NaCl.jl
#
# Edit `N_SUPER_LIST`, `H_VALUES`, `A_VALUES` below to broaden / narrow
# the sweep. Larger systems are slow on the *Ewald* reference (it's
# naive `O(N²·images)` real-space + `O(N·K)` reciprocal); the MSM side
# is cheap by comparison.

using MultilevelSummation
using MultilevelSummation.Tune: sweep, ewald_reference, SweepResult
using StaticArrays
using Printf

# ----------------------------------------------------------------------
# System construction
# ----------------------------------------------------------------------

const A_LAT = 4.0                              # NaCl-like lattice constant (Å)
                                                # nearest-neighbour distance = A_LAT/2 = 2.0

"""
    build_nacl(n_super; a_lat = A_LAT)

Build an `n_super × n_super × n_super` supercell of a rock-salt NaCl
crystal with charges ±1.

Returns `(positions, charges, cell, periodic)`.
"""
function build_nacl(n_super::Int; a_lat::Float64 = A_LAT)
    # Rock-salt: two interpenetrating FCC sublattices, 8 ions per conventional cell.
    na_frac = (SVector(0.0, 0.0, 0.0), SVector(0.5, 0.5, 0.0),
               SVector(0.5, 0.0, 0.5), SVector(0.0, 0.5, 0.5))
    cl_frac = (SVector(0.5, 0.0, 0.0), SVector(0.0, 0.5, 0.0),
               SVector(0.0, 0.0, 0.5), SVector(0.5, 0.5, 0.5))
    positions = SVector{3,Float64}[]
    charges   = Float64[]
    @inbounds for i in 0:n_super-1, j in 0:n_super-1, k in 0:n_super-1
        offset = SVector(Float64(i), Float64(j), Float64(k)) .* a_lat
        for p in na_frac
            push!(positions, p .* a_lat .+ offset)
            push!(charges, +1.0)
        end
        for p in cl_frac
            push!(positions, p .* a_lat .+ offset)
            push!(charges, -1.0)
        end
    end
    box = n_super * a_lat
    cell = SMatrix{3,3,Float64}(box * one(SMatrix{3,3,Float64}))
    periodic = (true, true, true)
    return positions, charges, cell, periodic
end

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Supercell sizes giving roughly doubling ion counts in the
# 100–14k range. Comment the larger ones out for a fast first run.
const N_SUPER_LIST = (3, 4, 5, 6, 8, 10, 12)
#                     ↑                       ↑
#                     N = 216                 N = 13824

# (h, a) sweep — in Å. Cubic basis is the only one shipped today.
const H_VALUES = (0.5, 1.0, 2.0)
const A_VALUES = (2.0, 4.0, 8.0)

# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
function run_sweep()
    rows = NamedTuple[]
    for n_super in N_SUPER_LIST
        positions, charges, cell, periodic = build_nacl(n_super)
        N   = length(positions)
        box = cell[1, 1]
        @printf "\n=========  n_super = %d, N = %d ions, box = %.2f Å  =========\n" n_super N box

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

csv_path = joinpath(@__DIR__, "tune_NaCl_results.csv")
open(csv_path, "w") do io
    println(io, "N,n_super,box,h,n_grid,a,L,rel_err,t_msm,t_ewald,U_ref")
    for r in rows
        @printf io "%d,%d,%.4f,%.4f,%d,%.4f,%d,%.6e,%.6e,%.6e,%.6e\n" r.N r.n_super r.box r.h r.n_grid r.a r.L r.rel_err r.t_msm r.t_ewald r.U_ref
    end
end
@info "wrote $(length(rows)) rows to $csv_path"
