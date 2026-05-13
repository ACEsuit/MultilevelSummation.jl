# Kernels

Pair kernels are translation-invariant, isotropic functions ``K : \mathbb{R}^D \to \mathbb{R}``.
Each kernel is a plain Julia type with two methods:

- `(K::MyKernel)(r::SVector{D,T})` — evaluate ``K(r)``.
- `grad(K::MyKernel, r::SVector{D,T})` — evaluate ``\nabla K(r)``.

The package currently ships two families of concrete kernels; the
interface is duck-typed, so user-defined kernels work without inheriting
from any abstract type.

```@docs
InversePower
RationalDecay
```

`Coulomb{T}` is provided as the alias `InversePower{1,T}`.

```@docs
MLSum.grad
```
