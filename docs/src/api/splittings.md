# Splittings

A splitting bundles the softening function and decomposition parameters
matched to a specific kernel. It splits ``K = K_0 + K_1 + \dots + K_L``
where ``K_0`` has compact support ``|r| \le a`` and each ``K_l`` is
smooth (and itself compactly supported for ``l < L``).

A splitting `s` must implement:

- `short_range(s, r)` → ``K_0(r)``
- `long_range_level(s, l, r)` → ``K_l(r)`` for ``l = 1, \dots, L-1``
- `top_level(s, r)` → ``K_L(r)``
- `short_range_grad`, `long_range_level_grad`, `top_level_grad`
- `requires_neutralising_background(s)` → `Bool`

```@docs
HardyC2Cubic
short_range
long_range_level
top_level
short_range_grad
long_range_level_grad
top_level_grad
requires_neutralising_background
level_scale
```
