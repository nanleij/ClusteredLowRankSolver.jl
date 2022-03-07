# ClusteredLowRankSolver

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://nanleij.github.io/ClusteredLowRankSolver.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nanleij.github.io/ClusteredLowRankSolver.jl/dev)
[![Coverage](https://codecov.io/gh/nanleij/ClusteredLowRankSolver.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nanleij/ClusteredLowRankSolver.jl)

## Clustered Low-Rank Semidefinite Programs
A clustered low-rank semidefinite program is a semidefinite program in equality form including free scalar variables, where the constraint matrices corresponding to positive semidefinite variables have a low-rank structure. The program is clustered in the sense that two constraints in different clusters do not use the same positive semidefinite matrix variables.
An example where such a clustered low-rank SDP appears is by sampling (low-rank) sums-of-squares constraints. In the interface we focus on this application.


## Installation
After installing Julia, run Julia and install the package with e.g.
```
]add ClusteredLowRankSolver
```
Press `backspace` to go back to the REPL from the package environment.

## Usage
Use the package with `using ClusteredLowRankSolver`. See the [documentation](https://nanleij.github.io/ClusteredLowRankSolver.jl/stable) for instructions on using the interface. Below we show how to model a small polynomial optimization problem.

To use `n` threads, start Julia with the option `-t n`.
On Windows using multiple threads may lead to crashes or wrong answers when using free variables.

## Examples
Consider the problem of finding the minimum of a univariate polynomial p(x) over [-1,1], i.e., the maximal λ such that p(x)-λ >=0. Relaxing the constraint gives p(x)-λ = s_1(x) + (1-x^2) * s_2(x) , i.e., s_1(x) + (1-x^2) * s_2(x) + λ = p where s_i are sum-of-squares polynomials.

```julia
using ClusteredLowRankSolver, AbstractAlgebra
using BasesAndSamples # To generate the samples

# Set up the polynomial space and define the polynomial to be optimized:
R, (x,) = PolynomialRing(RealField,["x"])
p = 1-x-x^3+x^6
# The degree of the basis for the sums-of-squares polynomials we want to use:
d = 3

# For the constraint we consider the part for the free variables and the part for the
# positive semidefinite matrix variables separately
free_dict = Dict(λ => 1)  

# The matrix variables come from the sum-of-squares parts, since s_i(x) is a sum-of-squares
# if and only if s_i(x) = ⟨b(x)b(x)^T, Y⟩ for some Y ⪰ 0 and vector of basis polynomials b(x)
matrix_dict = Dict()
# Both SOS parts have the same total degree, but different weights
matrix_dict[Block(:SOS1)] = LowRankMatPol([R(1)], [[x^k for k=0:d]])
matrix_dict[Block(:SOS2)] = LowRankMatPol([1-x^2], [[x^k for k=0:d-1]])

# Chebyshev points in [-1,1]:
samples = sample_points_chebyshev(2d)
# Create the constraint:
con = Constraint(p, matrix_dict, free_dict, samples)

# we want to maximize λ
obj = Objective(0, Dict(), Dict(λ => 1))

# Maximize the objective with constraint `con`
sos = LowRankSOSProblem(true, obj, [con])
# Convert the SOS problem to a clustered low-rank SDP
sdp = ClusteredLowRankSDP(sos)
# Solve the sdp with the standard parameters
status, result = solvesdp(sdp)
```

More examples are available in the [documentation](https://nanleij.github.io/ClusteredLowRankSolver.jl/stable). For the code examples described in the documentation and more, see the [examples](https://github.com/nanleij/ClusteredLowRankSolver.jl/tree/main/examples) folder.


## Citing ClusteredLowRankSolver
If you use `ClusteredLowRankSolver.jl` in a publication please consider citing

 - D. de Laat and N. Leijenhorst, *Solving clustered low-rank semidefinite programs arising from polynomial optimization*, Preprint (2022), [arXiv:2202.12077](https://arxiv.org/abs/2202.12077)
