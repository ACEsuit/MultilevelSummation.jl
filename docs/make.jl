using Documenter
using MLSum

DocMeta.setdocmeta!(MLSum, :DocTestSetup, :(using MLSum); recursive=true)

makedocs(;
    modules  = [MLSum],
    sitename = "MLSum.jl",
    authors  = "Christoph Ortner and contributors",
    remotes  = nothing,
    format = Documenter.HTML(;
        edit_link = "main",
        assets    = String[],
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home"            => "index.md",
        "Algorithm"       => "algorithm.md",
        "Examples"        => "examples.md",
        "API reference"   => [
            "Kernels"      => "api/kernels.md",
            "Splittings"   => "api/splittings.md",
            "Basis"        => "api/basis.md",
            "Grid"         => "api/grid.md",
            "Operators"    => "api/operators.md",
            "Reference"    => "api/reference.md",
            "Calculator"   => "api/calculator.md",
        ],
        "Internals notes" => "internals.md",
    ],
    warnonly = [:missing_docs, :cross_references],
)
