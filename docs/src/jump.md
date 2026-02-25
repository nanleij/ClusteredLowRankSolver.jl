# Using JuMP
Since version 2.0.0, ClusteredLowRankSolver supports [JuMP](https://jump.dev/JuMP.jl/stable/). Models should be created using
```julia
    model = GenericModel{BigFloat}(ClusteredLowRankSolver.Optimizer)
```

It is possible to round solutions from JuMP models using `find_field` and `exact_solution`. However, due to how JuMP is designed, the model will not be modified since the number type of the model cannot be changed to Nemo numbers. If no bridges are used (the model only uses PSD, nonnegative and free variables, and equality constraints), the variables in the `Problem` and `PrimalSolution` have the same names as the JuMP variables (where possible).  

Below is the code for the Lovasz-theta number of the 5-cycle, modified from the [JuMP documentation](https://jump.dev/JuMP.jl/stable/tutorials/conic/simple_examples/#Lovász-numbers).
```julia
using ClusteredLowRankSolver, JuMP
function example_theta_problem()
    # define the model
    model = GenericModel{BigFloat}(ClusteredLowRankSolver.Optimizer)
    # need precise enough solution for the rounding
    set_attribute(model, "duality_gap_threshold", 1e-30)

    # define the SDP
    E = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    @variable(model, X[1:5, 1:5], PSD)
    for i in 1:5
        for j in (i+1):5
            if !((i, j) in E || (j, i) in E)
                A = zeros(Int, 5, 5)
                A[i, j] = 1
                A[j, i] = 1
                @constraint(model, LinearAlgebra.dot(A, X) == 0)
            end
        end
    end
    @constraint(model, LinearAlgebra.tr(LinearAlgebra.I * X) == 1)
    J = ones(Int, 5, 5)
    @objective(model, Max, LinearAlgebra.dot(J, X))

    # solve the SDP
    optimize!(model)
    
    assert_is_solved_and_feasible(model)
    
    println("The Lovász number is: $(objective_value(model))")
    
    # round the solution
    FF, g = find_field(model)
    status, problem, esol = exact_solution(model; FF, g)
    # problem and esol are the ClusteredLowRankSolver structs, so we can extract the matrix using
    matrixvar(esol, :X)
    # If bridges where used to transform the problem in a SDP in equality form,
    # the names of the variables are indices (e.g. matrixvar(esol, 1))

    println("The exact objective is $(objvalue(problem, esol)) with z approximately equal to $g")
    return objvalue(problem, esol), g
end

```