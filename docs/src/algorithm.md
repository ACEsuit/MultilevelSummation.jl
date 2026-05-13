# The MSM algorithm

This page is a brief walk-through of the multilevel summation method as
implemented in `MultilevelSummation.jl`. The reference is Hardy *et al.* (2015); the
notation here follows the paper.

## Kernel splitting

For an isotropic, translation-invariant pair kernel ``K(r) = \kappa(|r|)``,
MSM decomposes the kernel into a short-range piece with compact support
``|r| \le a`` plus ``L`` slowly varying pieces:

```math
K = K_0 + K_1 + \dots + K_L .
```

For the Coulomb kernel ``\kappa(s) = 1/s`` and the paper's ``C^2`` cubic
softening function ``\gamma``, the splitting (eq. 2 of the paper) is

```math
\begin{aligned}
K_0(r) &= \frac{1}{|r|} - \frac{1}{a}\, \gamma(|r|/a), \\
K_l(r) &= \frac{1}{a_l}\,\gamma(|r|/a_l) - \frac{1}{a_{l+1}}\,\gamma(|r|/a_{l+1}),
  \quad l = 1, \dots, L-1, \\
K_L(r) &= \frac{1}{a_L}\,\gamma(|r|/a_L),
\end{aligned}
```

where ``a_l = 2^{l-1} a``. Each ``K_l`` with ``l \ge 1`` has compact
support ``|r| \le a_{l+1}``; ``K_L`` is the smooth tail.

In `MultilevelSummation.jl`:

| Object              | API                                  |
|---------------------|--------------------------------------|
| Softening ``\gamma``| `MultilevelSummation.gamma_softening(R)` (internal) |
| ``K_0(r)``          | `short_range(splitting, r)`          |
| ``K_l(r)``          | `long_range_level(splitting, l, r)`  |
| ``K_L(r)``          | `top_level(splitting, r)`            |

with the corresponding `_grad` variants returning ``\nabla K_l(r)``.

## Grid hierarchy

A nested hierarchy of grids is built. Level 1 has spacing ``h``; each
coarser level halves the resolution along every axis:

```
spacing:  h    2h   4h   ...   2^{L-1} h
size:     n    n/2  n/4  ...   1   (for periodic axes; open axes
                                    are simply halved)
```

For periodic axes the cell extent ``L_\alpha`` must be an integer
multiple of ``2^{L-1} h``; for open axes the level-1 extent is set from
the particle bounding box plus a basis-support pad and rounded up to a
multiple of ``2^{L-1}``.

`MultilevelSummation.build_grid_hierarchy(cell, periodic, positions, calc)` returns
this hierarchy as a `Vector{UniformGrid{D,T}}`.

## Algorithmic steps

```
particles                                                particles
   │                                                        ▲
   │ anterpolate  (eq. 7)                interpolate  (eq. 12)
   ▼                                                        │
  q^1                                                      e^1 = e^{1,cutoff} + prolong(e^2)
   │                                                        ▲
   │ restrict    (eq. 8)                  prolong    (eq. 11)
   ▼                                                        │
  q^2                                                      e^2 = e^{2,cutoff} + prolong(e^3)
   │                                                        ▲
   │ restrict                              prolong          │
   ▼                                                        │
   .                                                        .
   .                                                        .
  q^L  ─── top-level direct sum (eq. 10) ──► e^L
```

1. **Anterpolation** (`anterpolate!`): scatter point charges to level-1
   grid points using the basis ``\Phi``.
2. **Restriction** (`restrict!`): coarsen the level-`l` grid charge to
   level `l+1`. This is the same basis-product scatter applied to grid
   points rather than particles.
3. **Grid-cutoff convolution** (`grid_cutoff!`): for ``l = 1, \dots, L-1``,
   compute ``e^{l,\text{cutoff}}[m] = \sum_n K_l(r_m - r_n)\, q^l[n]``
   using a precomputed stencil (compact support).
4. **Top-level direct sum** (`top_level!`): for ``l = L``, no compact
   support — just sum over all grid pairs. For fully periodic axes the
   top grid is reduced to one point and (if the splitting requires it)
   zeroed via `apply_neutralising_background!` to implement the
   neutralising background.
5. **Prolongation** (`prolong!`): refine ``e^{l+1}`` to the level-`l`
   grid and add to ``e^{l,\text{cutoff}}``.
6. **Interpolation** (`interpolate!`): gather the level-1 grid potential
   onto the particles.

## Energy and forces

The energy is assembled as (paper eq. 5):

```math
U^{\text{MSM}}
  = \tfrac{1}{2} \sum_{i \ne j} q_i q_j\, K_0(r_{ij})
  + \tfrac{1}{2} \sum_i q_i\, e^{\text{long}}_i
  - \tfrac{1}{2}\, K_\text{long}(0) \sum_i q_i^2 ,
```

where the last term cancels the spurious self-interaction at ``r = 0``
introduced by the long-range pathway. In code, `kernel_self_value` sums
the long-range component kernels at zero displacement.

Forces are the analytical gradient (paper eq. 13):

```math
F_i = -\, q_i \sum_{j\ne i} q_j\, \nabla K_0(r_{ij})
      -\, q_i \sum_m e^1_m\, \nabla \phi_m(r_i) .
```

The first term is the direct short-range force; the second is computed
via `interpolate_grad!`, which gathers the gradient of the basis at
``r_i`` against the level-1 potential.

The self-correction term has no spatial gradient.

## Boundary conditions

Each axis is treated independently:

- **Open**: the grid is finite, contributions outside the grid are
  dropped.
- **Periodic**: indices wrap; the top level is reduced to a single
  point.

Mixed BC (e.g. slab geometry: periodic in `xy`, open in `z`) emerges
automatically from per-axis handling — no special code path.

## Complexity

For ``N`` particles with cutoff ``a`` and grid spacing ``h``,
each MSM evaluation is

```math
\mathcal{O}\!\left( N \cdot \left(\tfrac{a}{h}\right)^d
                  + N_\text{grid} \cdot \text{stencil}^d
                  + N_\text{grid,top}^2 \right) .
```

The first term is the short-range direct sum, the second the per-level
grid-cutoff convolution, the third the top-level direct sum. The
prototype is not performance-tuned; the implementation is "kernel-shaped"
so that the per-level loops can be ported to `KernelAbstractions.jl`
later.

## Approximation accuracy

The energy error scales as ``\mathcal{O}(h^p / a^{p+1})`` with ``p`` the
order of the interpolation basis. The default `CubicC1` basis has
``p = 3``, so the error is third order in `h` and fourth order in `a`.
At fixed ``h``, growing ``a`` reduces error monotonically (verified in
the test suite). Holding ``h/a`` fixed does *not* reduce error — both
parameters must be tuned.
