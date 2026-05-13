using Documenter
using MultilevelSummation

DocMeta.setdocmeta!(MultilevelSummation, :DocTestSetup,
                    :(using MultilevelSummation); recursive=true)

# Point Documenter at the GitHub repo so source links + deploydocs work.
# Change the org/repo here if the package moves elsewhere.
const REPO = Documenter.Remotes.GitHub("ACEsuit", "MultilevelSummation.jl")

makedocs(;
    modules  = [MultilevelSummation],
    sitename = "MultilevelSummation.jl",
    authors  = "Christoph Ortner and contributors",
    repo     = REPO,
    format = Documenter.HTML(;
        canonical = "https://ACEsuit.github.io/MultilevelSummation.jl",
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

# Skipped automatically when ENV["CI"] != "true" or the git ref is wrong.
deploydocs(;
    repo = "github.com/ACEsuit/MultilevelSummation.jl.git",
    devbranch = "main",
    push_preview = true,
)
