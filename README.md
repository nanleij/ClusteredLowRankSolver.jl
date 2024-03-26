# ClusteredLowRankSolver

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nanleij.github.io/ClusteredLowRankSolver.jl/dev)
[![Coverage](https://codecov.io/gh/nanleij/ClusteredLowRankSolver.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nanleij/ClusteredLowRankSolver.jl)

# ClusteredLowRankSolver.jl

[`ClusteredLowRankSolver.jl`](https://github.com/nanleij/ClusteredLowRankSolver.jl) implements 
  - a primal-dual interior point method for solving semidefinite programming problems;
  - a minimal interface to model semidefinite programming problems including optional polynomial constraints; 
  - functionality for working with sampled polynomials; and
  - an implementation of a rounding heuristic which can round the numerical output of the solver to an exact optimal solution over rational or algebraic numbers. 
The solver can exploit the low-rank structure of constraint matrices (which arise naturally from enforcing polynomial identities by evaluating both sides at a unisolvent set) but can also work with dense constraint matrices. The solver uses high-precision numerics (using Arblib) and the interface integrates with the Nemo computer algebra system.

## Installation

The solver is written in Julia, and has been registered as a Julia package. Typing `using ClusteredLowRankSolver` in the REPL will prompt installation if the package has not been installed yet.

## Examples

Here we give two small examples to showcase the interface. More examples are available in the [documentation](https://nanleij.github.io/ClusteredLowRankSolver.jl/dev). For the code examples described in the documentation and more, see the [examples](https://github.com/nanleij/ClusteredLowRankSolver.jl/tree/main/examples) folder.

### Example 1: The Goemans-Williamson MAX-CUT relaxation
Given a Laplacian `L` of a graph with `n` vertices, the semidefinite programming relaxation of the max-cut problem reads
```math
\begin{aligned}
& \text{maximize} & & \left\langle \frac{1}{4}L, X \right\rangle\\
& \text{subject to} & & \langle E_{ii}, X \rangle = 1, \; i=1,\ldots,n,\\
&&&X \in S_+^{n}.
\end{aligned}
```
where ``E_{ii}`` is the matrix with a one in the ``(i,i)`` entry, and zeros elsewhere. Here ``\langle \cdot, \cdot \rangle`` denotes the trace inner product.

The following code implements this using `ClusteredLowRankSolver`.

```julia
using  ClusteredLowRankSolver
function goemans_williamson(L::Matrix)
    n = size(L, 1)

    # Construct the objective
    obj = Objective(0, Dict(:A => 1//4 * L), Dict())

    # Construct the constraints
    constraints = []
    for i = 1:n
        M = zeros(Rational{BigInt}, n, n)
        M[i, i] = 1//1
        # the first argument is the right hand side
        push!(constraints, Constraint(1, Dict(:A => M), Dict()))
    end

    # Construct the problem
    problem = Problem(Maximize(obj), constraints)

    # Solve the problem
    status, primalsol, dualsol, time, errorcode = solvesdp(problem)

    return objvalue(problem, dualsol), matrixvar(dualsol, :A)
end
```

For a three-cycle, this gives
```julia
L = [2 -1 -1; -1 2 -1; -1 -1 2]
obj, X = goemans_williamson(L)
obj # = 2.24999999999999972300549056142031245384884141740800
```

### Example 2: Finding the global minimum of a univariate polynomial
To find the minimum of a polynomial ``f`` of degree ``2d``, one can use the following problem
```math
\begin{aligned}
& \text{minimize} & & \lambda\\
& \text{subject to} & & f - \lambda = s,\\
\end{aligned}
```
where ``s`` is a sum-of-squares polynomial of degree ``2d``.
Let ``m`` be a vector whose entries form a basis of the polynomials up to degree ``d``, then we can write ``s = \langle m(x)m(x)^T, X \rangle`` where ``X`` is a positive semidefinite matrix.

```julia
using ClusteredLowRankSolver, Nemo
function polyopt(f, d)
    #set up the polynomial field 
    P = parent(f)
    u = gen(P)

    #compute the sos basis and the samples
    sosbasis = basis_chebyshev(d, u)
    samples = sample_points_chebyshev(2d,-1,1) 


    #construct the constraint SOS + lambda = f
    c = Dict()
    c[:X] = LowRankMatPol([1], [sosbasis[1:d+1]])
    constraint = Constraint(f, c, Dict(:lambda => 1), samples)

    #Construct the objective
    objective = Objective(0, Dict(), Dict(:lambda => 1))

    #Construct the SOS problem: minimize the objective s.t. the constraint
    problem = Problem(true, objective, [constraint])

    #Solve the SDP and return results
    status, primalsol, dualsol, time, errorcode = solvesdp(problem)    
    return objvalue(problem, dualsol)
end
```
Then we can for example find the minimum of the polynomial ``x^2+1`` using
```julia
R, x = polynomial_ring(QQ, :x)
minvalue = polyopt(x^2+1, 1)
```

### Exact version of Example 2
To find the minimum exactly we can use the following function.
```julia
using ClusteredLowRankSolver, Nemo

function polyopt_exact(f, d)
    # Set up the polynomial ring 
    P = parent(f)
    u = gen(P)

    # Compute the polynomial basis and the samples
    sosbasis = basis_chebyshev(d, u)
    samples = [round(BigInt, 10000x)//10000 for x in sample_points_chebyshev(2d, -1, 1)] 

    # Construct the constraint SOS + lambda = f
    c = Dict()
    c[:X] = LowRankMatPol([1], [sosbasis[1:d+1]])
    constraint = Constraint(f, c, Dict(:lambda => 1), samples)

    # Construct the objective
    objective = Objective(0, Dict(), Dict(:lambda => 1))

    # Construct the SOS problem: minimize the objective s.t. the constraint holds
    problem = Problem(Maximize(objective), [constraint])

    #Solve the SDP and return results
    status, primalsol, dualsol, time, errorcode = solvesdp(problem)    
    
    success, esol = exact_solution(problem, primalsol, dualsol)

    success, objvalue(problem, esol)
end
```
Then we can find the exact minimum of the polynomial ``x^2+1`` using
```julia
R, x = polynomial_ring(QQ, :x)
polyopt_exact(x^2+1, 1)
```

## Citing ClusteredLowRankSolver and the rounding procedure
The semidefinite programming solver and the interface (including sampled polynomials) in `ClusteredLowRankSolver.jl` have been developed as part of the paper
 - Nando Leijenhorst and David de Laat, [*Solving clustered low-rank semidefinite programs arising from polynomial optimization*](https://arxiv.org/abs/2202.12077), preprint, 2022. arXiv:2202.12077

The solver was inspired by the more specialized solver
- David Simmons-Duffin. [*A semidefinite program solver for the conformal bootstrap*](https://link.springer.com/article/10.1007/JHEP06(2015)174). J. High Energy Phys. 174 (2015), [arXiv:1502.02033](https://arxiv.org/abs/1502.02033)

The rounding procedure in `ClusteredLowRankSolver.jl` has been developed as part of the paper
 - Henry Cohn, David de Laat, and Nando Leijenhorst, [*Optimality of spherical codes via exact semidefinite programming bounds*](), preprint, 2024. arXiv:2403.16874

This improves the rounding procedure developed in
- Maria Dostert, David de Laat, and Philippe Moustrou, [*Exact semidefinite programming bounds for packing problems*](https://epubs.siam.org/doi/10.1137/20M1351692), SIAM J. Optim. 31(2) (2021), 1433-1458, [arXiv:2001.00256](https://arxiv.org/abs/2001.00256)
