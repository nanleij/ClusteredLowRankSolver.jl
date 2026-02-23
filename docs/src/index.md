# ClusteredLowRankSolver.jl

[`ClusteredLowRankSolver.jl`](https://github.com/nanleij/ClusteredLowRankSolver.jl) implements 
  - a primal-dual interior point method for solving semidefinite programming problems;
  - a minimal interface to model semidefinite programming problems with (optional) polynomial equality constraints; 
  - functionality for working with sampled polynomials; and
  - an implementation of a rounding heuristic which can round the numerical output of the solver to an exact optimal solution over rational or algebraic numbers. 
The solver can exploit the low-rank structure of constraint matrices (which arise naturally from enforcing polynomial identities by evaluating both sides at a unisolvent set) but can also work with dense constraint matrices. The solver uses [Arb](https://arblib.org) for high-precision numerics and the interface integrates with the [Nemo](https://nemocas.github.io/Nemo.jl/stable/) computer algebra system.

## Installation

The solver is written in Julia, and has been registered as a Julia package. Typing `using ClusteredLowRankSolver` in the REPL will prompt installation if the package has not been installed yet.

## Examples

Here we give two small examples to showcase the interface. For more explanation on the interface, see the [Tutorial](@ref). 

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

```@example 1
using ClusteredLowRankSolver

function goemans_williamson(L::Matrix; eps=1e-40)
    n = size(L, 1)

    # Construct the objective
    obj = Objective(0, Dict(:X => 1//4 * L), Dict())

    # Construct the constraints
    constraints = []
    for i = 1:n
        M = zeros(Rational{BigInt}, n, n)
        M[i, i] = 1//1
        # the first argument is the right hand side
        push!(constraints, Constraint(1, Dict(:X => M), Dict()))
    end

    # Construct the problem: Maximize the objective s.t. the constraints hold
    problem = Problem(Maximize(obj), constraints)

    # Solve the problem
    status, dualsol, primalsol, time, errorcode = solvesdp(problem, duality_gap_threshold=eps)

    objvalue(problem, primalsol), matrixvar(primalsol, :X)
end
nothing # hide
```
For a three-cycle, this gives
```@example 1
L = [2 -1 -1; -1 2 -1; -1 -1 2]
obj, X = goemans_williamson(L)
obj
```

### Example 2: Finding the global minimum of a univariate polynomial
To find the minimum of a polynomial ``f`` of degree ``2d``, one can use the following problem
```math
\begin{aligned}
& \text{maximize} & & \lambda\\
& \text{subject to} & & f - \lambda = s,\\
\end{aligned}
```
where ``s`` is a sum-of-squares polynomial of degree ``2d``.
Let ``m`` be a vector whose entries form a basis of the polynomials up to degree ``d``, then we can write ``s = \langle m(x)m(x)^T, X \rangle``, where ``X`` is a positive semidefinite matrix.

```@example 2
using ClusteredLowRankSolver, Nemo

function polyopt(f, d)
    # Set up the polynomial ring 
    P = parent(f)
    u = gen(P)

    # Compute the polynomial basis and the samples
    sosbasis = basis_chebyshev(d, u)
    samples = sample_points_chebyshev(2d, -1, 1) 

    # Construct the constraint SOS + lambda = f
    c = Dict()
    c[:X] = LowRankMatPol([1], [sosbasis[1:d+1]])
    constraint = Constraint(f, c, Dict(:lambda => 1), samples)

    # Construct the objective
    objective = Objective(0, Dict(), Dict(:lambda => 1))

    # Construct the SOS problem: minimize the objective s.t. the constraint holds
    problem = Problem(Maximize(objective), [constraint])

    #Solve the SDP and return results
    status, dualsol, primalsol, time, errorcode = solvesdp(problem)    
    
    objvalue(problem, primalsol)
end
nothing # hide
```
Then we can for example find the minimum of the polynomial ``x^2+1`` using
```@example 2
R, x = polynomial_ring(QQ, :x)
minvalue = polyopt(x^2+1, 1)
```

### [Exact version of Example 2](@id rounding_univariate)
To find the minimum exactly we can use the following function.
```@example 2
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
    status, dualsol, primalsol, time, errorcode = solvesdp(problem)    
    
    success, esol = exact_solution(problem, dualsol, primalsol)

    success, objvalue(problem, esol)
end
nothing # hide
```
Then we can find the exact minimum of the polynomial ``x^2+1`` using
```@example 2
R, x = polynomial_ring(QQ, :x)
polyopt_exact(x^2+1, 1)
```
