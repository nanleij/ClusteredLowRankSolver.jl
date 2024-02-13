using Documenter
push!(LOAD_PATH,"../src/")
using ClusteredLowRankSolver
using DocumenterCitations
using Bibliography

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

sort_bibliography!(bib.entries, :nyt)

makedocs(plugins=[bib],
    sitename = "ClusteredLowRankSolver.jl Documentation",
    format = Documenter.HTML(),
    # modules = [ClusteredLowRankSolver], #we don't want all the doc strings in the documentation
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Interface" => "manual/interface.md",
            "manual/sampling_clustering.md",
            "manual/solver.md",
        ],
        "Examples" => [
            "Delsarte LP bound" =>"examples/delsarte.md",
            "Sphere packing" => "examples/sphere_packing.md",
            "Symmetric polynomial optimization" =>"examples/poly_opt.md",
            "Clustering" => "examples/clustering.md"
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
