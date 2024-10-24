# Using SDPA-sparse format

The function [`sdpa_sparse_to_problem`](@ref) can be used to read an SDPA-sparse file and create a [`Problem`](@ref). In particular, this makes it possible to use [JuMP](https://jump.dev/) to create a semidefinite program and [write it to the SDPA-sparse format](https://jump.dev/JuMP.jl/stable/manual/models/#Write-a-model-to-file), which can then be read and solved with `ClusteredLowRankSolver`. The JuMP interface is not yet supported for direct conversion to a [`Problem`](@ref).

```@docs
sdpa_sparse_to_problem
```

Since the default formulation in JuMP is different from the formulation in ClusteredLowRankSolver, it is recommended to use the package [`Dualization.jl`](https://github.com/jump-dev/Dualization.jl) before writing the problem to a file in JuMP.
```julia
using Dualization

model = Model()
% code to make the model

write_to_file(dualize(model), "sdp.dat-s")
```