# Grid operators

The four primary operators that move information through the MSM
pipeline. All are in-place (`!`) and dimension-generic.

## Particles ↔ grid

```@docs
anterpolate!
interpolate!
interpolate_grad!
```

## Coarser ↔ finer grid

```@docs
restrict!
prolong!
```

## Kernel application on the grid

```@docs
grid_cutoff!
build_stencil
top_level!
apply_neutralising_background!
top_grid_size
```
