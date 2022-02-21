module ClusteredLowRankSolver

using AbstractAlgebra, IterTools, LinearAlgebra
# using ApproximateFekete
using Printf, Arblib, BlockDiagonals
using KrylovKit

export LowRankMat, LowRankMatPol, Block, Constraint, Objective,	 LowRankPolProblem, ClusteredLowRankSDP, solvesdp, approximatefekete,SampledMPolyElem,optimal
export solvesdp, convert_to_prec, SolverFailure

import LinearAlgebra: dot, transpose
import Base: ==

import AbstractAlgebra: evaluate

include("interface.jl")
include("tools.jl")
include("threadinginfo.jl")
include("solver.jl")
include("approximate_fekete.jl") #I think this is better than having another auxiliary package

end
