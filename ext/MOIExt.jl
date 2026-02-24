module MOIExt
# MOI wrapper

using ClusteredLowRankSolver
import MathOptInterface as MOI
using LinearAlgebra

function __init__()
    setglobal!(ClusteredLowRankSolver, :Optimizer, Optimizer)
    return
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    # before solving:
    initiated::Bool
    problem::Problem
    options::Dict{Symbol, Any}
    block_dims::Vector{Int}
    variable_map::Vector{Union{Int, Tuple{Int,Int}, Tuple{Int, Int, Int}}} # Variable Index vi -> blk, i, j, like SDPA
    #after solving
    optimized::Bool
    result_data::Dict{Any, Any}
    # add default options to options
    function Optimizer()
        opt = new()
        opt.options = Dict(:prec=>precision(BigFloat), # The precision used in the algorithm.
            :maxiterations=>500,
            :beta_infeasible=>3//10, # try to decrease mu by this factor when infeasible
            :beta_feasible => 1//10, # try to decrease mu by this factor when feasible
            :gamma=>9//10, # this fraction of the maximum possible step size is used
            :omega_p=>big(10)^(10), # initial size of the dual PSD variables #TODO: define one omega to be the same as the other?
            :omega_d=>big(10)^(10), # initial size of the primal PSD variables
            :duality_gap_threshold=>1e-15, # how near to optimal does the solution need to be
            :dual_error_threshold=>1e-30,  # how feasible does the dual solution need to be
            :primal_error_threshold=>1e-30, # how feasible does the primal solution need to be
            :max_complementary_gap=>big(10)^100, # the maximum of <X,Y>/#rows(X)
            :need_dual_feasible=>false, # terminate when the solution is dual feasible
            :need_primal_feasible=>false, # terminate when the solution is primal feasible
            :verbose => true, # false: print nothing, true: print information after each iteration
            :step_length_threshold=>1e-7,
            :safe_step=>true,
            :correctoronly=>false,
            :save_settings=>SaveSettings()) 
        opt.initiated = false
        opt.optimized = false
        opt.result_data = Dict{Any, Any}()
        opt.variable_map = Vector{Union{Int, Tuple{Int,Int}, Tuple{Int, Int, Int}}}()
        opt.block_dims = Int[]
        return opt
    end
end

function MOI.is_empty(opt::Optimizer)
    return !opt.initiated
end

function MOI.empty!(opt::Optimizer)
    opt.initiated = false
    opt.optimized = false
    empty!(opt.variable_map)
    empty!(opt.result_data)
    empty!(opt.block_dims)
    return 
end

function Base.summary(io::IO, opt::Optimizer)
    if opt.initiated
        return print(io, 
    """ClusteredLowRankSolver with $(length(opt.problem.constraints)) constraints, 
    $(length(opt.result_data[:primalsol].freevars)) free variables and 
    $(length(opt.result_data[:primalsol].matrixvars)) psd variables ($(sum(length(opt.block_dims))) entries).
    """)
    else 
        return print(io, """Uninitiated ClusteredLowRankSolver""")
    end
end

# get set and supports for :
# RelativeGapTolerance
# Silent
# TimeLimitSec
# SolverName
# SolverVersion # version of julia package, so not needed?

function MOI.get(opt::Optimizer, attr::MOI.SolverName)
    return "ClusteredLowRankSolver"
end

function MOI.get(opt::Optimizer, attr::MOI.SolverVersion)
    return string(pkgversion(ClusteredLowRankSolver))
end

function MOI.get(opt::Optimizer, attr::Union{MOI.RelativeGapTolerance, MOI.NumberOfThreads})
    if attr isa MOI.RelativeGapTolerance
        return opt.options[:duality_gap_threshold]
    elseif attr isa MOI.NumberOfThreads
        return Threads.nthreads()
    end
    error("Attribute not recognized")
end


function MOI.supports(opt::Optimizer, attr::Union{MOI.RelativeGapTolerance, MOI.NumberOfThreads})
    return true
end

function MOI.set(opt::Optimizer, attr::MOI.RelativeGapTolerance, val)
    opt.options[:duality_gap_threshold] = val
end

function MOI.supports(opt::Optimizer, attr::MOI.RawOptimizerAttribute)
    return haskey(opt.options, Symbol(attr.name))
end

function MOI.get(opt::Optimizer, attr::MOI.RawOptimizerAttribute)
    if !MOI.supports(opt, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    return opt.options[Symbol(attr.name)]
end

function MOI.set(opt::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(opt, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    return opt.options[Symbol(attr.name)] = value
end

function MOI.get(opt::Optimizer, attr::MOI.RawSolver) 
    # Would want to return Problem, but that gives an error. This is what they expect?
    return opt
end

function MOI.get(opt::Optimizer, ::MOI.Silent)::Bool
    return !opt.options[:verbose]
end
function MOI.set(opt::Optimizer, ::MOI.Silent, s::Bool)::Nothing
    opt.options[:verbose] = !s
    return 
end

MOI.supports(opt::Optimizer, ::MOI.Silent) = true






# Support of constraints
# function MOI.supports_constraint(
#     ::Optimizer,
#     ::Type{MOI.}, # function or union of functions
#     ::Type{MOI.}, # set or union of sets
# )
#     return true
# end
const _SupportedSets =
    Union{MOI.Nonnegatives, MOI.PositiveSemidefiniteConeTriangle}

function MOI.supports_add_constrained_variables(
    ::Optimizer,
    ::Type{<:_SupportedSets},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{MOI.EqualTo{T}},
) where T <: Real
    return true
end

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
    },
) where T<:Real
    return true
end



# optimize call(s)
# copy_to(dest::Optimizer, src::MOI.ModelLike)
# optimize!(m::Optimizer)

# using a similar structure as the code of SDPA.jl
function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    # copy the model src to our format
    MOI.empty!(dest)
    index_map = MOI.Utilities.IndexMap()
    # Make the index map: 
    #1) for the nonnegative/PSD variables
    _constrain_variables_on_creation(dest, src, index_map, MOI.Nonnegatives)
    _constrain_variables_on_creation(
        dest,
        src,
        index_map,
        MOI.PositiveSemidefiniteConeTriangle,
    )
    #2) for the free variables
    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    k = 1 # free variable index
    for vi_src in vis_src
        if !haskey(index_map, vi_src)
            # add variable
            push!(dest.variable_map, k)
            index_map[vi_src] = MOI.VariableIndex(length(dest.variable_map))
            k+=1
        end
    end
    # make the constraints
    cons = []
    k = 1
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if S <: _SupportedSets
            continue # constrainted on creation
        end
        if !MOI.supports_constraint(dest, F, S)
            throw(MOI.UnsupportedConstraint{F, S}())
        end
        cis_src = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        for (k_intern, ci_src) in enumerate(cis_src)
            func = MOI.get(src, MOI.CanonicalConstraintFunction(), ci_src)
            set = MOI.get(src, MOI.ConstraintSet(), ci_src)
            if !iszero(MOI.constant(func))
                throw(
                    MOI.ScalarFunctionConstantNotZero{<:Real,F,S}(
                        MOI.constant(func),
                    ),
                )
            end
            psd_dict = Dict()
            free_dict = Dict()
            for term in func.terms
                if !iszero(term.coefficient)
                    idx = dest.variable_map[index_map[term.variable].value]
                    if length(idx) > 1
                        if length(idx) == 2
                            blk = idx[1]
                            i = j = 1
                        else
                            blk, i, j = idx
                        end
                        if !haskey(psd_dict, blk)
                            psd_dict[blk] = zeros(typeof(term.coefficient), dest.block_dims[blk], dest.block_dims[blk])
                        end
                        psd_dict[blk][i,j] += term.coefficient/2
                        psd_dict[blk][j,i] += term.coefficient/2
                    else
                        # free variable
                        free_dict[idx] = term.coefficient
                    end
                end
            end
            if isempty(psd_dict)
                throw(MOI.AddConstraintNotAllowed{F,S}("ClusteredLowRankSolver only support constraints including PSD or Nonnegative variables"))
            end
            push!(cons, Constraint(MOI.constant(set), psd_dict, free_dict))
            index_map[ci_src] = MOI.ConstraintIndex{F,S}(k)
            k+=1 # actually count which constraint it is
        end
    end
    if isempty(cons)
        error("ClusteredLowRankSolver requires at least one constraint")
    end

    # create the objective
    maximize = true
    constant = 0
    psd_dict = Dict()
    free_dict = Dict()
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr == MOI.Name()
            continue
        elseif attr == MOI.ObjectiveSense()
            maximize = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        elseif attr isa MOI.ObjectiveFunction
            F = MOI.get(src, MOI.ObjectiveFunctionType())
            if !(F <: MOI.ScalarAffineFunction{<:Real})
                error("objective function type $F not supported")
            end
            obj = MOI.Utilities.canonical(MOI.get(src, MOI.ObjectiveFunction{F}()))
            for term in obj.terms
                if !iszero(term.coefficient)
                    idx = dest.variable_map[index_map[term.variable].value]
                    if length(idx) > 1
                        if length(idx) == 2
                            blk = idx[1]
                            i = j = 1
                        else
                            blk, i, j = idx
                        end
                        if !haskey(psd_dict, blk)
                            psd_dict[blk] = zeros(typeof(term.coefficient), dest.block_dims[blk], dest.block_dims[blk])
                        end
                        psd_dict[blk][i,j] += term.coefficient/2
                        psd_dict[blk][j,i] += term.coefficient/2
                    else
                        # free variable
                        free_dict[idx] = term.coefficient
                    end
                end
            end
            constant = obj.constant
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    obj = Objective(constant, psd_dict, free_dict)

    # create the problem
    dest.problem = Problem(maximize, obj, cons)

    #Q: how to load the options (default or non default?)
    # or is that done by JuMP somehow through the get and set methods?

    dest.initiated = true
    return index_map
end


function _constrain_variables_on_creation(
    dest::MOI.ModelLike,
    src::MOI.ModelLike,
    index_map::MOI.Utilities.IndexMap,
    ::Type{S},
) where {S<:MOI.AbstractVectorSet}
    for ci_src in
        MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,S}())
        f_src = MOI.get(src, MOI.ConstraintFunction(), ci_src)
        if !allunique(f_src.variables) # psd constraint with constraints between the entries?
            _error(
                "Cannot copy constraint `$(ci_src)` as variables constrained on creation because there are duplicate variables in the function `$(f_src)`",
                "to bridge this by creating slack variables.",
            )
        elseif any(vi -> haskey(index_map, vi), f_src.variables) #psd variable with entries which are already variables
            _error(
                "Cannot copy constraint `$(ci_src)` as variables constrained on creation because some variables of the function `$(f_src)` are in another constraint as well.",
                "to bridge constraints having the same variables by creating slack variables.",
            )
        else
            set = MOI.get(src, MOI.ConstraintSet(), ci_src)::S
            vis_dest, ci_dest = _add_constrained_variables(dest, set)
            index_map[ci_src] = ci_dest
            for (vi_src, vi_dest) in zip(f_src.variables, vis_dest)
                index_map[vi_src] = vi_dest
            end
        end
    end
    return
end
function _add_constrained_variables(optimizer::Optimizer, set::_SupportedSets)
    offset = length(optimizer.variable_map)
    _new_block(optimizer, set)
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables,typeof(set)}(offset + 1)
    return [MOI.VariableIndex(i) for i in offset .+ (1:MOI.dimension(set))], ci
end

function _error(start, stop)
    return error(
        start,
        ". Use `MOI.instantiate(ClusteredLowRankSolver.Optimizer; with_bridge_type = Float64)` ",
        stop,
    )
end

function _new_block(optimizer::Optimizer, set::MOI.Nonnegatives)
    blk = length(optimizer.block_dims)
    for i in 1:MOI.dimension(set)
        push!(optimizer.block_dims, 1)
        push!(optimizer.variable_map, (blk+i, MOI.dimension(set)))
    end
    return
end

function _new_block(
    optimizer::Optimizer,
    set::MOI.PositiveSemidefiniteConeTriangle,
)
    push!(optimizer.block_dims, set.side_dimension)
    blk = length(optimizer.block_dims)
    for i in 1:set.side_dimension
        for j in 1:i
            push!(optimizer.variable_map, (blk, i, j))
        end
    end
    return
end


function MOI.optimize!(opt::Optimizer)
    status, dualsol, primalsol, t, e = solvesdp(opt.problem; opt.options...)
    opt.optimized = true
    pr = opt.problem
    opt.result_data[:primalsol] = primalsol
    opt.result_data[:dualsol] = dualsol
    opt.result_data[:status] = status
    opt.result_data[:errorcode] = e
    opt.result_data[MOI.SolveTimeSec()] = t
    opt.result_data[MOI.ObjectiveValue()] = objvalue(pr, primalsol)
    opt.result_data[MOI.DualObjectiveValue()] = pr.objective.constant + (-1)^(!pr.maximize) * sum(dualsol.x[i][1] * pr.constraints[i].constant for i in eachindex(dualsol.x))
    return 
end


# some required attributes: (get)
# PrimalStatus
# DualStatus
# RawStatusString
# ResultCount
# TerminationStatus

function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    # result: ::MOI.ResultStatusCode
    if !opt.optimized || attr.result_index > 1
        return MOI.NO_SOLUTION
    end
    status = opt.result_data[:status]
    if typeof(status)<:Union{Optimal, PrimalFeasible, Feasible, NearOptimal}
        return MOI.FEASIBLE_POINT
    else
        return MOI.INFEASIBLE_POINT
    end
end
function MOI.get(opt::Optimizer, attr::MOI.DualStatus)
    # result: ::MOI.ResultStatusCode
    if !opt.optimized || attr.result_index > 1
        return MOI.NO_SOLUTION
    end
    status = opt.result_data[:status]
    if typeof(status)<:Union{Optimal, DualFeasible, Feasible, NearOptimal}
        return MOI.FEASIBLE_POINT
    else
        return MOI.INFEASIBLE_POINT
    end
end
function MOI.get(opt::Optimizer, ::MOI.RawStatusString)
    if !opt.optimized
        return error("`MOI.optimize!` not called.")
    end
    return string(opt.result_data[:status])
    # result: MOI.String
end


function MOI.get(opt::Optimizer, attr::Union{MOI.ObjectiveValue, MOI.DualObjectiveValue})
    MOI.check_result_index_bounds(opt, attr)
    return opt.result_data[attr]
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    if !opt.optimized
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status = opt.result_data[:status]
    e = opt.result_data[:errorcode]
    if status isa Optimal
        return MOI.OPTIMAL
    elseif e == 2
        return MOI.ITERATION_LIMIT
    elseif e == 3 # infeasible or unbounded because large duality gap
        if abs(MOI.get(opt, MOI.ObjectiveValue())) > 1e30 && !(status isa DualFeasible) && !(status isa NearOptimal) && !(status isa Feasible)
            # probably unbounded
            return MOI.DUAL_INFEASIBLE
        elseif abs(MOI.get(opt, MOI.DualObjectiveValue())) > 1e30 && !(status isa PrimalFeasible) && !(status isa NearOptimal) && !(status isa Feasible)
            #probably infeasible
            return MOI.INFEASIBLE
        end
        return MOI.INFEASIBLE_OR_UNBOUNDED
    elseif e == 4 #small step length, so probably not primal-dual feasible?
        if abs(MOI.get(opt, MOI.ObjectiveValue())) > 1e30 && !(status isa DualFeasible) && !(status isa NearOptimal) && !(status isa Feasible)
            # probably unbounded
            return MOI.DUAL_INFEASIBLE
        elseif abs(MOI.get(opt, MOI.DualObjectiveValue())) > 1e30 && !(status isa PrimalFeasible) && !(status isa NearOptimal) && !(status isa Feasible)
            #probably infeasible
            return MOI.INFEASIBLE
        end
        return MOI.SLOW_PROGRESS
    elseif e == 1
        # we don't distinguish between numerical errors and interrupt
        return MOI.OTHER_ERROR 
    end
end
function MOI.get(opt::Optimizer, ::MOI.ResultCount)::Int
    return opt.optimized ? 1 : 0
end

# ObjectiveValue
# SolveTimeSec
# VariableDual
# ConstraintPrimal
# PrimalObjectiveValue
# VariableDualStart
# VariablePrimalStart

function MOI.get(opt::Optimizer, attr::MOI.SolveTimeSec)
    if opt.optimized
        return opt.result_data[attr]
    end
    error("`MOI.optimize!` not called.")
end
    

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, attr)
    idx = opt.variable_map[vi.value]
    if length(idx) == 3
        blk, i, j = idx
        return opt.result_data[:primalsol].matrixvars[blk][i,j]
    elseif length(idx) == 2 # 1x1 matrix variables
        blk, numvars = idx
        return opt.result_data[:primalsol].matrixvars[blk][1,1]
    else
        #free variable
        return opt.result_data[:primalsol].freevars[idx]
    end
end

function MOI.get(opt::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}) where T<:Real
    MOI.check_result_index_bounds(opt, attr)
    return -opt.result_data[:dualsol].x[ci.value][1]
end

function MOI.get(opt::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives})
    MOI.check_result_index_bounds(opt, attr)
    blk_start, numvars = opt.variable_map[ci.value]
    return [opt.result_data[:dualsol].matrixvars[blk_start+i][1,1] for i=0:numvars-1]
end

function MOI.get(opt::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle})
    MOI.check_result_index_bounds(opt, attr)
    blk, i, j = opt.variable_map[ci.value]
    blkdim = opt.block_dims[blk]
    return [opt.result_data[:dualsol].matrixvars[blk][i,j] for i=1:blkdim for j=1:i]
end

function MOI.get(opt::Optimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Nonnegatives})
    MOI.check_result_index_bounds(opt, attr)
    blk, numvars = opt.variable_map[ci.value]
    return [opt.result_data[:primalsol].matrixvars[blk+i][1,1] for i=0:numvars-1]
end

function MOI.get(opt::Optimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.PositiveSemidefiniteConeTriangle})
    MOI.check_result_index_bounds(opt, attr)
    blk, i, j = opt.variable_map[ci.value]
    blkdim = opt.block_dims[blk]
    return [opt.result_data[:primalsol].matrixvars[blk][i,j] for j=1:blkdim for i=1:j]
end

function MOI.get(opt::Optimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}) where T<:Real
    MOI.check_result_index_bounds(opt, attr)
    # find the value of <A, X>
    con = opt.problem.constraints[ci.value] 
    return (sum(dot(con.matrixcoeff[k], opt.result_data[:primalsol].matrixvars[k]) for k in eachindex(con.matrixcoeff); init=T(0))
        + sum(con.freecoeff[k] * opt.result_data[:primalsol].freevars[k] for k in eachindex(con.freecoeff); init=T(0)))
end

end
