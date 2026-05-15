# Common scaffolding for the KernelAbstractions-backed code path:
#
# - `_resolve_backend`: pick the execution path. The kwarg wins; otherwise
#   array type is consulted; otherwise return `nothing` to mean "legacy
#   OhMyThreads CPU path".
# - `_ka_zeros`: thin wrapper over `KernelAbstractions.zeros` so the
#   orchestrator can stay backend-generic.
# - Internal "force the KA path" stubs (`_msm_compute_ka`, ...). They
#   `error` until the `*_ka.jl` files supply real methods.
#
# The point of the indirection is that `MSMCalculator`'s type stays
# unchanged: the GPU/KA path is selected per-call from the arrays (or the
# explicit `backend` kwarg), not from a calculator field. This sidesteps
# the constructor-collision concern with T2 in PRIORITIES.md.

using KernelAbstractions: KernelAbstractions, Backend
using GPUArraysCore: AbstractGPUArray

"""
    _resolve_backend(positions, charges, backend_kwarg) -> Backend or nothing

Pick the execution backend for a call to `msm_energy` / `msm_energy_forces`:

1. If the caller passed `backend = <something>` explicitly, use it.
2. Else if either array argument is an `AbstractGPUArray`, read the
   backend from it via `KernelAbstractions.get_backend`.
3. Else return `nothing`, meaning "fall through to the legacy
   OhMyThreads CPU path".

Returning `nothing` is the load-bearing third state — it lets callers
keep the existing CPU implementation completely unchanged when no KA
arrays are involved.
"""
@inline _resolve_backend(positions, charges, backend_kwarg) =
    backend_kwarg !== nothing            ? backend_kwarg :
    positions  isa AbstractGPUArray      ? KernelAbstractions.get_backend(positions) :
    charges    isa AbstractGPUArray      ? KernelAbstractions.get_backend(charges)   :
    nothing

"""
    _ka_zeros(backend, ::Type{T}, dims...) -> AbstractArray{T}

Allocate a zero-initialised array on `backend`. Thin wrapper so callers
read closer to the existing `zeros(T, ...)` idiom.
"""
@inline _ka_zeros(backend::Backend, ::Type{T}, dims::Integer...) where {T} =
    KernelAbstractions.zeros(backend, T, dims...)

"""
    _ka_synchronize(backend) -> nothing

Call `KernelAbstractions.synchronize(backend)` if a method exists,
otherwise no-op. Some backends (notably `JLArrays`'s `JLBackend`) don't
define `synchronize` because their kernel launches are already
synchronous — calling KA's generic `synchronize` on them would
`MethodError`. Catching that specific error keeps real-GPU backends
(CUDA, ROCm, …) unaffected.
"""
@inline function _ka_synchronize(backend)
    try
        KernelAbstractions.synchronize(backend)
    catch e
        e isa MethodError || rethrow()
    end
    return nothing
end

# ----- KA-path entry points: declared here, implemented in src/*_ka.jl ------
#
# Until the rest of the `*_ka.jl` files are wired up, calling these is an
# error. The dispatch shim in `_msm_compute` only reaches them when a
# non-`nothing` backend has been resolved, so they cannot fire on the
# default CPU path.

function _msm_compute_ka end
function _short_range_and_long_forces_ka end
