using JuMP, ClusteredLowRankSolver, LinearAlgebra

# the SDP formulations are examples from the JuMP documentation
function example_theta_problem()
    model = GenericModel{BigFloat}(ClusteredLowRankSolver.Optimizer)
    set_attribute(model, "duality_gap_threshold", 1e-30)
    # set_silent(model)
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
    optimize!(model)
    
    assert_is_solved_and_feasible(model)
    
    println("The Lovász number is: $(objective_value(model))")
    
    # round the solution
    FF, g = find_field(model)
    status, problem, esol = exact_solution(model; FF, g)

    println("The exact objective is $(objvalue(problem, esol)) with z approximately equal to $g")
    return objvalue(problem, esol), g
end

function example_POVM()
    N, d = 2,2
    states = [1//2 * [1, -1] * [1, -1]', 1//2 * [1, -im] * [1, -im]']

    model = GenericModel{BigFloat}(ClusteredLowRankSolver.Optimizer)
    set_attribute(model, "duality_gap_threshold", 1e-30)

    E = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for i in 1:N]
    @constraint(model, sum(E) == LinearAlgebra.I)

    @objective(model, Max, sum(real(LinearAlgebra.dot(states[i], E[i])) for i in 1:N) / N)

    optimize!(model)

    FF, g = find_field(model)
    status, problem, esol = exact_solution(model; FF, g)
    println("The exact objective is $(objvalue(problem, esol)) with z approximately equal to $g")
    return objvalue(problem, esol), g
end
