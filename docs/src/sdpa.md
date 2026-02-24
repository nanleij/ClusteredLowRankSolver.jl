# Using SDPA-sparse format

The function [`sdpa_sparse_to_problem`](@ref) can be used to read an SDPA-sparse file and create a [`Problem`](@ref). Since v2.0.0, it is possible to build and solve problems through [JuMP](https://jump.dev/JuMP.jl/stable/).

```@docs
sdpa_sparse_to_problem
```

If this is used in combination with JuMP, it is recommended to use the package [`Dualization.jl`](https://github.com/jump-dev/Dualization.jl) before writing the problem to a file in JuMP.
```julia
using Dualization

model = Model()
% code to make the model

write_to_file(dualize(model), "sdp.dat-s")
```