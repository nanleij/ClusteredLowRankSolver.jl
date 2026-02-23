import MathOptInterface as MOI

# test with BigFloat since we return BigFloat
model = MOI.instantiate(ClusteredLowRankSolver.Optimizer; with_bridge_type = BigFloat)
MOI.Test.runtests(model, MOI.Test.Config(BigFloat, rtol=1e-10, atol=1e-10,exclude=Any[MOI.VariableName, MOI.ConstraintName, MOI.delete,  MOI.ConstraintBasisStatus, MOI.ObjectiveBound]),
    exclude=[
    # adds 1 free variable, no constraints. That is not supported by ClusteredLowRankSolver (and doesn't make sense to do in real applications either)
    "test_attribute_RawStatusString", 
    "test_attribute_SolveTimeSec", 
    "test_model_copy_to_UnsupportedAttribute",
    "test_objective_ObjectiveFunction_blank",
    "test_solve_TerminationStatus_DUAL_INFEASIBLE",
    # gets reformulated so that there is no constraints (except >=0 )
    "test_DualObjectiveValue_Max_VariableIndex_LessThan", 
    "test_DualObjectiveValue_Min_VariableIndex_GreaterThan", 
    # supposed to fail with coefficient type UInt8 (but why?)
    "test_model_supports_constraint_ScalarAffineFunction_EqualTo",
    # Float/BigFloat error due to RSOC to SOC bridge (I think). Hard to track down where exactly the Float is introduced
    "test_conic_HermitianPositiveSemidefiniteConeTriangle_1", 
    "test_conic_SecondOrderCone_VectorOfVariables", 
    "test_constraint_PrimalStart_DualStart_SecondOrderCone", 
    # no constraint on a variable. We remove the variable (with warning), they want dual_infeasible
    "test_conic_SecondOrderCone_no_initial_bound", 
    # linearly dependent variables/constraints: not (yet) supported
    "test_linear_transform",
    ],
)

function test_attribute_leftovers()   
    model = ClusteredLowRankSolver.Optimizer()

    # set a status and pretend the model is optimized
    model.result_data[MOI.SolveTimeSec()] = 0.1
    model.result_data[:status] = ClusteredLowRankSolver.Optimal()
    model.optimized = true
    model.initiated = true
    @test MOI.get(model, MOI.RawStatusString()) isa MOI.attribute_value_type(MOI.RawStatusString())
    @test MOI.get(model, MOI.SolveTimeSec()) isa Float64

    return
end

test_attribute_leftovers()

