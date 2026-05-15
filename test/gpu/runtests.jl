# Developer / GPU-CI entry point for the CPU↔GPU equivalence check.
#
# Not run as part of the standard `]test` target. Use:
#
#   julia --project=test/gpu -e 'using Pkg; Pkg.develop(path=".."); Pkg.add("CUDA")'
#   julia --project=test/gpu test/gpu/runtests.jl
#
# Replace `CUDA` with `AMDGPU`, `Metal`, or `oneAPI` to match your
# hardware. The first one of those found in the project's *direct*
# dependencies is used.

using Pkg
using Test

const project_deps = Set(keys(Pkg.project().dependencies))

const has_cuda   = "CUDA"   in project_deps
const has_amdgpu = "AMDGPU" in project_deps
const has_metal  = "Metal"  in project_deps
const has_oneapi = "oneAPI" in project_deps

# Use a top-level if/elseif chain (`using` and runtime evaluation
# don't mix well — dynamic `@eval using ...` triggers world-age
# errors when the freshly-loaded module is read in the same function
# scope). The chain is parsed once; only one `using` actually fires.

if has_cuda
    using CUDA
    const FRAMEWORK     = "CUDA"
    const BACKEND       = CUDABackend()
    const MAKE_GPU_ARR  = CuArray
    const FUNCTIONAL    = CUDA.functional()
elseif has_amdgpu
    using AMDGPU
    const FRAMEWORK     = "AMDGPU"
    const BACKEND       = ROCBackend()
    const MAKE_GPU_ARR  = ROCArray
    const FUNCTIONAL    = AMDGPU.functional()
elseif has_metal
    using Metal
    const FRAMEWORK     = "Metal"
    const BACKEND       = MetalBackend()
    const MAKE_GPU_ARR  = MtlArray
    const FUNCTIONAL    = Metal.functional()
elseif has_oneapi
    using oneAPI
    const FRAMEWORK     = "oneAPI"
    const BACKEND       = oneAPIBackend()
    const MAKE_GPU_ARR  = oneArray
    const FUNCTIONAL    = oneAPI.functional()
else
    @info """
    No supported GPU framework was found in the `test/gpu/` environment.

    To run the equivalence tests, add one of the following to this project:
        julia --project=test/gpu -e 'using Pkg; Pkg.add("CUDA")'
        julia --project=test/gpu -e 'using Pkg; Pkg.add("AMDGPU")'
        julia --project=test/gpu -e 'using Pkg; Pkg.add("Metal")'
        julia --project=test/gpu -e 'using Pkg; Pkg.add("oneAPI")'
    """
    exit(0)
end

if !FUNCTIONAL
    @info "$(FRAMEWORK) is installed but not functional on this host. Exiting cleanly."
    exit(0)
end

@info "Running CPU↔GPU equivalence tests on $(FRAMEWORK) ($(typeof(BACKEND)))"

include("equivalence.jl")
@testset "CPU ↔ $FRAMEWORK equivalence" begin
    run_equivalence_tests(BACKEND, MAKE_GPU_ARR)
end
