# ClusteredLowRankSolver.jl

[`ClusteredLowRankSolver.jl`](https://github.com/nanleij/ClusteredLowRankSolver.jl) provides a primal-dual interior point method for solving clustered low-rank semidefinite programs. This can be used for (semidefinite) programs with polynomial inequality constraints, which can be rewritten in terms of sum-of-squares polynomials. See the [manual]() for a detailed description of the problems and the usage of the solver.

## Citation

If you use `ClusteredLowRankSolver.jl` in work that results in a publication, consider citing
 - D. de Laat and N.M. Leijenhorst, *Solving Clustered Low-Rank Semidefinite Programs arising from Polynomial Optimization*, arXiv:

## Clustered Low-Rank Semidefinite Programs

```math
\begin{aligned}
	\min \quad & \sum_j \langle Y^j, C^j \rangle + \langle y, b\rangle \\
	\text{s.t.} \quad & \langle Y^j, A^j_* \rangle + B^T y = c \\
	& Y^j \succeq 0
\end{aligned}
```

## Documentation

```@autodocs
Modules = [ClusteredLowRankSolver]
```
