module ClusteredLowRankSolver

using IterTools, LinearAlgebra, Printf, BlockDiagonals, GenericLinearAlgebra
using Nemo
using Arblib
using Combinatorics: multiexponents
using Serialization

const AL = Arblib
const RF = Nemo.AbstractAlgebra.RealField # Nemo realfield is ArbField(64), this one is BigFloats (264 bits)

using KrylovKit

using Random #for randomly choosing columns in the rounding procedure
using RowEchelon #for detecting kernel vectors in rounding.jl

# import AbstractAlgebra
import LinearAlgebra: dot, transpose, issymmetric
import Base: ==
import Nemo: evaluate

export LowRankMatPol, Block, Constraint, Objective,	Problem, ClusteredLowRankSDP, solvesdp, approximatefekete, optimal
export name
export Maximize, Minimize

export objective, matrixcoeff, freecoeff, matrixcoeffs, freecoeffs, constraints

export approximatefeketeexact

export generic_embedding, basic_embedding

export convert_to_prec, SolverFailure
export check_sdp!, check_problem
export SampledMPolyRingElem, SampledMPolyRing, sampled_polynomial_ring
export model_psd_variables_as_free_variables


export DualSolution, PrimalSolution, matrixvar, freevar, slacks, matrixvars, freevars

export vectorize, as_primal_solution, blocksizes, addconstraint!

export linearsystem, objvalue, partial_linearsystem
export RoundingSettings, linearsystem_coefficientmatching
#Q: what do we want to export here? I imagine that people will mostly use exact_solution, not the basis_transformations etc
# export basis_transformations, detecteigenvectors, transform, undo_transform, project_to_affine_space
export exact_solution
export is_valid_solution

export find_field, to_field

export basis_monomial, basis_laguerre, basis_jacobi, basis_chebyshev, basis_gegenbauer
export sample_points_simplex, sample_points_padua, sample_points_rescaled_laguerre, sample_points_chebyshev, sample_points_chebyshev_mod

export sdpa_sparse_to_problem

export SaveSettings

include("interface.jl")

include("tools.jl")

include("threadinginfo.jl")
include("solver.jl")
include("approximate_fekete.jl") #I think this is better than having another auxiliary package
include("checks.jl")

include("rounding.jl")
include("find_field.jl")

include("basesandsamples.jl")

include("SDPAtoCLRS.jl")

include("MOI_wrapper/MOI_wrapper.jl")

# include("precompile.jl")



end
