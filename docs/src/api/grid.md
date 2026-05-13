# Grid

The MSM grid hierarchy is built from `UniformGrid` instances — one per
level. Each grid is described by its per-axis spacing, extent, origin,
and per-axis boundary conditions.

```@docs
UniformGrid
grid_zeros
npoints
wrap_index
particle_to_grid
coarser_grid
build_grid_hierarchy
```
