module JuMPExt
# find_field and exact_solution with JuMP models

using ClusteredLowRankSolver
using JuMP: GenericModel, solver_name, get_attribute, object_dictionary, MOI


"""
    exact_solution(model::GenericModel{T}; FF=QQ, g=BigFloat(1), settings=RoundingSettings(), transformed=false, verbose=true)

Round an approximate solution of a JuMP model for ClusteredLowRankSolver to an exact solution. The function
returns whether the rounding was successful, the problem, and the solution. If transformed=true, the function
also returns transformations: the solution is of the form BXB^T with X positive definite in the solution struct, 
and B in a dictionary. If no bridges are used, the variable names are the same as in the JuMP model.
That is, @variable(model, X[1:2, 1:2], PSD) will be referred to in the returned problem and solution as :X.
If there are bridges used, the function returns the problem as used by the solver and an exact solution to this problem (if successful)
"""
function ClusteredLowRankSolver.exact_solution(model::GenericModel{T}; FF=QQ, g=BigFloat(1), settings::RoundingSettings=RoundingSettings(), transformed=false, verbose=true) where T
    # We cannot modify the model so that value(X) (if X is the JuMP variable) returns exact numbers. 
    # They expect type T <: Real, and since the solver returns BigFloat, the default needs to be T=BigFloat
    # Also, Nemo's numbers are not subtypes of Real, so we cannot make a GenericModel{QQFieldElem}, for example.
    # So, the best we can do is to return the exact problem together with an exact solution.
    # The main downside is that it does not correspond directly to the JuMP model/variables if they use bridges
    # and the names are not directly clear (but I guess that it is just numbered in the order the variables are introduced)
    # Q: can we get use variable names?
    @assert solver_name(model) == "ClusteredLowRankSolver"
    opt = get_attribute(model, MOI.RawSolver())
    @assert opt.optimized
    problem = map(x->generic_embedding(x, g; base_ring=FF), opt.problem)
    # map to the 'jump' variable names instead of indices
    idx_to_var = Dict()
    try
        objs = object_dictionary(model)
        for (k,x) in objs
            if length(x) > 1
                idx = opt.variable_map[x[1].index.value]
            else
                idx = opt.variable_map[x.index.value]
            end
            if length(idx) > 1
                idx_to_var[idx[1]] = k
            else
                idx_to_var[idx] = k
            end
        end   
    catch
    end
    
    if transformed
        success, esol, transformations = exact_solution(problem, opt.result_data[:dualsol], opt.result_data[:primalsol]; FF, g, settings, transformed, verbose)
        try
            problem = change_varkeys(problem, idx_to_var)
            esol = change_varkeys(esol, idx_to_var)
            transformations = Dict(idx_to_var[i]=>m for (i,m) in transformations)
        catch 
        end
        return success, problem, esol, transformations
    else
        success, esol = exact_solution(problem, opt.result_data[:dualsol], opt.result_data[:primalsol]; FF, g, settings, transformed, verbose)
        try
            problem = change_varkeys(problem, idx_to_var)
            esol = change_varkeys(esol, idx_to_var)
        catch 
        end
        return success, problem, esol
    end
end

function change_varkeys(p::Problem, idx_to_var)
    cs = [change_varkeys(c, idx_to_var) for c in p.constraints]
    return Problem(p.maximize, change_varkeys(p.objective, idx_to_var), cs)
end

function change_varkeys(c::Constraint, idx_to_var)
    dct_mats = Dict(idx_to_var[k]=>m for (k,m) in matrixcoeffs(c))
    dct_fv = Dict(idx_to_var[k]=>m for (k,m) in freecoeffs(c))
    return Constraint(c.constant, dct_mats, dct_fv, c.samples, c.scalings)
end

function change_varkeys(c::Objective, idx_to_var)
    dct_mats = Dict(idx_to_var[k]=>m for (k,m) in matrixcoeffs(c))
    dct_fv = Dict(idx_to_var[k]=>m for (k,m) in freecoeffs(c))
    return Objective(c.constant, dct_mats, dct_fv)
end

function change_varkeys(c::Union{PrimalSolution{T}, DualSolution{T}}, idx_to_var) where T
    dct_mats = Dict(idx_to_var[k]=>m for (k,m) in matrixvars(c))
    if c isa PrimalSolution
        dct_fv = Dict(idx_to_var[k]=>m for (k,m) in freevars(c))
        return PrimalSolution{T}(c.base_ring, dct_mats, dct_fv)
    else
        return DualSolution{T}(c.base_ring, c.x, dct_mats)
    end
end

function ClusteredLowRankSolver.find_field(model::GenericModel{T}, max_degree=10; valbound=1e-15,errbound=1e-15, bits=max_degree*100, max_coeff=10^5) where T
    @assert solver_name(model) == "ClusteredLowRankSolver"
    opt = get_attribute(model, MOI.RawSolver())
    find_field(opt.result_data[:dualsol], opt.result_data[:primalsol], max_degree; valbound, errbound, bits, max_coeff)
end

end


