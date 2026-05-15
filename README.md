# MultilevelSummation.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/MultilevelSummation.jl/dev/)
[![CI](https://github.com/ACEsuit/MultilevelSummation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ACEsuit/MultilevelSummation.jl/actions/workflows/CI.yml)

A Julia implementation of the **Multilevel Summation Method (MSM)** of
Hardy et al. (*J. Chem. Theory Comput.* **11**, 766–779, 2015) for fast
evaluation of long-range pair interactions, with support for any
dimension `d ∈ {1, 2, 3}`, per-axis `:open` / `:periodic` boundary
conditions, pluggable kernel splittings and interpolation bases, and an
[AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) +
[AtomsCalculators](https://github.com/JuliaMolSim/AtomsCalculators.jl)
interface. Ships with a naive 3D Ewald reference
(`MultilevelSummation.Reference`) and a programmatic hyperparameter
sweep API (`MultilevelSummation.Tune`) for accuracy-vs-cost analysis.

**Highly experimental** 
— the API is not stable
- only the Coulomb (`1/r`) splitting is currently implemented
- provides forces, but no ChainRules integration yet

The default CPU path uses
[OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl); launch
Julia with `julia -t auto` (or set `JULIA_NUM_THREADS`) for parallelism.
A second implementation path is available through
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
selected per call from the input array type or via a
`backend = <KA backend>` kwarg on `msm_energy` /
`msm_energy_forces`. The KA path covers every hot operator (anter- /
interpolation, restriction / prolongation, grid-cutoff convolution,
top-level direct sum) plus the short-range pair sum, the latter via
[NeighbourLists.jl](https://github.com/JuliaMolSim/NeighbourLists.jl)'s
GPU-friendly cell list. Pass `AbstractGPUArray` positions / charges
(e.g. `CuArray`s) and the backend is auto-detected; pass `backend =
KA.CPU()` to validate the KA path on plain `Array`s without GPU
hardware.

### CPU ↔ GPU equivalence test

The standalone script at [`test/gpu/runtests.jl`](test/gpu/runtests.jl)
runs an equivalence check (CPU result vs both the kwarg-driven and the
array-type-dispatched KA paths) on small NaCl and H2O fixtures. It is
deliberately kept out of the standard `]test` target so the heavy
binary deps of CUDA / AMDGPU / Metal / oneAPI are not pulled into
normal CI installs. To run it, add whichever framework matches your
hardware to `test/gpu/`:

```julia-repl
julia> using Pkg
julia> Pkg.develop(path=".")                      # cd'd into test/gpu first
julia> Pkg.add("CUDA")                            # or AMDGPU / Metal / oneAPI
```

Then `julia --project=test/gpu test/gpu/runtests.jl`. The script
auto-detects the installed framework and exits cleanly with a help
message if none is found.

See the [documentation](https://ACEsuit.github.io/MultilevelSummation.jl/dev/) 
for details, examples, and the implementation plan.
