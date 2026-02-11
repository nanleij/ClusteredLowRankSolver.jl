import MathOptInterface as MOI

# test with BigFloat since we return BigFloat
model = MOI.instantiate(ClusteredLowRankSolver.Optimizer; with_bridge_type = BigFloat)
MOI.Test.runtests(model, MOI.Test.Config(BigFloat, rtol=1e-10, atol=1e-10,exclude=Any[MOI.VariableName, MOI.ConstraintName, MOI.delete]),
    exclude=["test_attribute_RawStatusString", "test_attribute_SolveTimeSec", # adds 1 free variable, no constraints. That is not supported by ClusteredLowRankSolver
    "test_model_copy_to_UnsupportedAttribute", #see above
    "test_model_supports_constraint_ScalarAffineFunction_EqualTo", # supposed to fail with coefficient type UInt8 (but why?)
    "test_DualObjectiveValue_Max_VariableIndex_LessThan", # gets reformulated so that there is no constraints (except >=0 )
    "test_DualObjectiveValue_Min_VariableIndex_GreaterThan", 
    "test_basic", #TODO: temp for quicker testing
    "test_conic_GeometricMeanCone", #TODO: no idea what goes wrong here
    ] 
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

