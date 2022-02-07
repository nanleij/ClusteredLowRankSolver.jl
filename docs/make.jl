using Documenter
push!(LOAD_PATH,"../src/")
using ClusteredLowRankSolver

makedocs(
    sitename = "ClusteredLowRankSolver.jl Documentation",
    format = Documenter.HTML(),
    modules = [ClusteredLowRankSolver],
    pages = ["ClusteredLowRankSolver.jl" => "index.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nanleij/ClusteredLowRankSolver.jl.git",
    devbranch="main"
)
