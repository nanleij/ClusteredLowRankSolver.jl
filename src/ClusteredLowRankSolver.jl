module ClusteredLowRankSolver

using IterTools, LinearAlgebra, Printf, BlockDiagonals, GenericLinearAlgebra
using Nemo
using Arblib
const AL = Arblib
using KrylovKit

using Random #for randomly choosing columns in the rounding procedure
using RowEchelon #for detecting kernel vectors in rounding.jl

# import AbstractAlgebra
import LinearAlgebra: dot, transpose
import Base: ==
import Nemo: evaluate

export LowRankMatPol, Block, Constraint, Objective,	Problem, ClusteredLowRankSDP, solvesdp, approximatefekete, optimal

export approximatefekete_exact

export convert_to_prec, SolverFailure
export check_sdp!, check_problem
export SampledMPolyRingElem, SampledMPolyRing
export model_psd_variables_as_free_variables!

export PrimalSolution, DualSolution, matrixvar, freevar, objective, slacks

export vectorize, as_dual_solution, blocksizes, addconstraint!

export linearsystem, objvalue, partial_linearsystem
export RoundingSettings, linearsystem_coefficientmatching

export find_field

include("interface.jl")

include("tools.jl")

include("threadinginfo.jl")
include("solver.jl")
include("approximate_fekete.jl") #I think this is better than having another auxiliary package
include("checks.jl")

include("rounding.jl")
include("find_field.jl")

#TODO figure out what to do here
# include("examples/maxcut.jl")
# include("examples/theta.jl")

end
