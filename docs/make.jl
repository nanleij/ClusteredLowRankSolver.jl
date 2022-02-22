using Documenter
push!(LOAD_PATH,"../src/")
using ClusteredLowRankSolver
using DocumenterCitations

bib = CitationBibliography("references.bib", sorting=:nyt)

makedocs(bib,
    sitename = "ClusteredLowRankSolver.jl Documentation",
    format = Documenter.HTML(),
    modules = [ClusteredLowRankSolver],
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Interface" => "manual/interface.md",
            "manual/solver.md",
        ],
        "Examples" => [
            "Delsarte LP bound" =>"examples/delsarte.md",
            "Sphere packing" => "examples/sphere_packing.md",
            "Symmetric polynomial optimization" =>"examples/poly_opt.md",
        ],
        "references.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nanleij/ClusteredLowRankSolver.jl.git",
    devbranch="main"
)
