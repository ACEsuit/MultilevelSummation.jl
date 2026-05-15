using Test

@testset "MultilevelSummation.jl" begin
    include("test_kernels.jl")
    include("test_basis.jl")
    include("test_splittings.jl")
    include("test_naive.jl")
    include("test_ewald.jl")
    include("test_anterp.jl")
    include("test_transfer.jl")
    include("test_gridcutoff.jl")
    include("test_toplevel.jl")
    include("test_core.jl")
    include("test_calculator.jl")
    include("test_systems.jl")
    include("test_tune.jl")
    include("test_ka_gridcutoff.jl")
    include("test_ka_transfer.jl")
    include("test_ka_toplevel.jl")
    include("test_ka_interp.jl")
    include("test_ka_anterp.jl")
    include("test_ka_shortrange.jl")
    include("test_ka_core.jl")
end
