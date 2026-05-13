# Interpolation basis

The basis ``\Phi(\xi)`` is the 1-D dimensionless interpolation function;
the ``D``-dimensional basis values are tensor products of ``\Phi`` along
each axis. A basis `b` must implement:

- `support_radius(b)` → half-width in grid spacings (integer).
- `eval_phi(b, ξ)` → ``\Phi(\xi)``.
- `eval_phi_prime(b, ξ)` → ``\Phi'(\xi)``.

```@docs
CubicC1
eval_phi
eval_phi_prime
support_radius
```
