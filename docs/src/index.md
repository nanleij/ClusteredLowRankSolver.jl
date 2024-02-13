# ClusteredLowRankSolver.jl

[`ClusteredLowRankSolver.jl`](https://github.com/nanleij/ClusteredLowRankSolver.jl) provides a primal-dual interior point method for solving clustered low-rank semidefinite programs. This can be used for (semidefinite) programs with polynomial inequality constraints, which can be rewritten in terms of sum-of-squares polynomials.

## Clustered Low-Rank Semidefinite Programs
A clustered low-rank semidefinite program is defined as
```math
\begin{aligned}
	\min \quad &a +  \sum_j \langle Y^j, C^j \rangle + \langle y, b\rangle \\
	\text{s.t.} \quad & \langle Y^j, A^j_* \rangle + B^T y = c \\
	& Y^j \succeq 0,
\end{aligned}
```
where the optimization is over the positive semidefinite matrices ``Y^j`` and the vector of free variables ``y``. Here ``\langle Y^j, A^j_*\rangle `` denotes the vector with entries ``\langle Y^j, A^j_p\rangle`` and the matrices ``A^j_p`` have the structure
```math
	A_p^j = \bigoplus_{l=1}^{L_j} \sum_{l=1}^{L_j} \sum_{r,s=1}^{R_j(l)} A_p^j(l;r, s) \otimes E_{r,s}^{R_j(l)}.
```
For fixed ``j, l``, the matrices ``A_p^j(l;r,s)`` are either normal matrices or of low rank. Furthermore, the full matrices are symmetric, which translates to the condition ``A_p^j(l;r, s)^{\sf T} =  A_p^j(l;s, r)``. The matrix ``E_{r,s}^n`` is the ``n \times n`` matrix with a one at position ``(r,s)`` and zeros otherwise. 

One example where this structure shows up is when using polynomial constraints which are converted to semidefinite programming constraints by sampling.
Such a semidefinite program with low-rank polynomial constraints is defined as
```math
\begin{aligned}
	\min \quad & a + \sum_j \langle Y^j, C^j \rangle + \langle y, b\rangle \\
	\text{s.t.} \quad & \langle Y^j, A^j_*(x) \rangle + B^T(x) y = c(x) \\
	& Y^j \succeq 0,
\end{aligned}
```
where ``A^j_p(x)`` have the same structure as before but have now polynomials as entries. This can be obtained from polynomial inequality constraints by sum-of-squares characterizations. The interface currently focusses on such semidefinite programs with polynomial constraints.

The implementations contains data types for a clustered low-rank semidefinite programs (`ClusteredLowRankSDP`) and semidefinite programs with low-rank polynomial constraints (`LowRankPolProblem`), where a `LowRankPolProblem` can be converted into a `ClusteredLowRankSDP`. The implementation also contains data types for representing low-rank (polynomial) matrices as well as functions and data types for working with samples and sampled polynomials. 

## Installation
The solver is written in Julia, and has been registered as a Julia package. Typing `using ClusteredLowRankSolver` in the REPL will prompt installation if the package has not been installed yet (from Julia 1.7 onwards).


## Documentation

### Solver
```@docs
solvesdp
```

### Interface
```@docs
ClusteredLowRankSDP
LowRankPolProblem
Objective
Constraint
LowRankMatPol
Block
approximatefekete
check_problem
check_sdp!
```

