# Tune

The `MultilevelSummation.Tune` submodule provides programmatic
hyperparameter sweeps. Given a system and an absolute reference
energy (e.g. from
[`MultilevelSummation.Reference`](reference.md)), it iterates over
`(h, a, L)` combinations and returns a table of
[`SweepResult`](@ref) rows so the user can inspect them and pick
parameters in their own code.

```julia
using MultilevelSummation
using MultilevelSummation.Tune

U_ref = Tune.ewald_reference(positions, charges, cell)
rows  = Tune.sweep(positions, charges, cell, periodic;
                   reference  = U_ref,
                   h_values   = (0.5, 1.0, 2.0),
                   a_values   = (2.0, 4.0, 8.0),
                   L_strategy = :all)

best  = Tune.recommend(rows; max_rel_err = 1e-3)
front = Tune.pareto_front(rows)
```

The shipped `tuning/` scripts (currently `tune_NaCl.jl`, with
`tune_H2O.jl` planned) are thin callers of the same API on
realistic test systems; their CSV output is the canonical way to
explore accuracy-vs-cost trade-offs on a given problem class.

## Result type

```@docs
MultilevelSummation.Tune.SweepResult
```

## Sweep + helpers

```@docs
MultilevelSummation.Tune.sweep
MultilevelSummation.Tune.pareto_front
MultilevelSummation.Tune.recommend
MultilevelSummation.Tune.ewald_reference
```
