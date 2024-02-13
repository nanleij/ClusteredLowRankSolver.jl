# The solver
`ClusteredLowRankSolver` is a primal-dual interior point method. Besides the dual program, it simultaneously solves the primal program
```math
\begin{aligned}
  \max & \sum_{j=1}^J \langle c^j, x^j \rangle &\\
  \text{s.t.} & \sum_{j=1}^J (B^j)^{\sf T} x^j = b &\\
   & X^{j}= \sum_{p=1}^{P_j} x_p^j A_p^{j} - C^{j} \succeq 0,\quad &j=1,\ldots,J.
\end{aligned}
```
Starting from a (probably infeasible) solution, the solver iteratively improves the solution using Newton steps until it is sufficiently close to being feasible and optimal. For a discussion of the general algorithm, see [simmons-duffin-semidefinite-2015](@cite); for the modifications needed to be able to use the general low-rank structures and for a discussion of the parallelization strategy, see [de-laat-clustered-2022](@cite). The feasibility is measured by the violation of the primal and dual constraints given by
```math
\begin{aligned}
    p &= \sum_{j=1}^J (B^j)^{\sf T} x^j - b, \\
    P^j &= X^{j} - \sum_{p=1}^{P_j} x_p^j A_p^{j} + C^{j}, \\
    d^j &= \big\langle A_*^{j}, Y^{j}\big\rangle + B^jy - c^j.
\end{aligned}
```
We denote the primal and dual errors by
```math
  ε_p = \max(\max_{r}( |p_r|), \max_{j,r,s}(|P^j_{rs}|))
```
and
```math
    ε_d = \max_{j,r}( |d^j_r|),
```
respectively. A solution ``(x,X,y,Y)`` is primal (dual) feasible if the primal (dual) error is at most the threshold `primal_error_threshold` (`dual_error_threshold`). A solution is feasible if it is both primal and dual feasible.

Denoting the primal objective by ``O_p(x,X)`` and the dual objective by ``O_d(y,Y)``, a measure of optimality is given by the (normalized) duality gap
```math
    Δ = \frac{|O_p(x,X) - O_d(y,Y)|}{\max(1,|O_p(x,X) + O_d(y,Y) |)}.
```
which measures the relative difference between the objectives of the primal and the dual problem.
A solution is considered optimal if it is feasible and if the duality gap is at most `duality_gap_threshold`.
The solver iteratively improves the feasibility and the duality gap of the solution.

## Options
Here we list the most important options. For the remaining options, see the documentation and the explanation of the algorithm in [simmons-duffin-semidefinite-2015](@cite).
  -  `prec` - The number of bits used for the calculations. The default is the `BigFloat` precision, which defaults to 256 bits.
  - `duality_gap_threshold` - Gives an indication of how close the solution is to the optimal solution. As a rule of thumb, a duality gap of ``10^{-(k+1)}`` gives ``k`` correct digits.  Default: ``10^{-15}``
  - `gamma` - The step length reduction; if a step of ``\alpha`` is possible, a step of ``\min(\gamma \alpha, 1)`` is taken. A lower `gamma` results in a more stable convergence, but can be significantly slower. Default: ``0.9``.
  - `omega_p`, `omega_d` - The size of the initial primal respectively dual solution. A low `omega` can keep the solver from converging, but a high `omega` in general increases the number of iterations needed and thus also the solving time. Default: ``10^{10}``
  - `need_primal_feasible`, `need_dual_feasible` - If `true`, terminate when a primal or dual feasible solution is found, respectively. Default: `false`.

## Output
When the option `verbose` is `true` (default), the solver will output information for every iteration.
In order of output, we have (where the values are from the start of the iteration except for the step lengths, which are only known at the end of the iteration)
  - The iteration number
  - The time since the start of the first iteration
  - The complementary gap ``\mu = \langle X, Y \rangle / K`` where ``K`` is the number of rows of ``X``. The solution will converge to the optimum for ``\mu \to 0``.
  - The primal objective ``O_p(y,Y)``
  - The dual objective ``O_d(y,Y)``
  - The relative duality gap ``Δ``
  - The primal matrix error ``\max_{j,r,s}|P^j_{rs}|``.
  - The primal scalar error ``\max_{r}|p_r|``
  - The dual (scalar) error ``ε_d``
  - The primal step length
  - The dual step length
  - ``\beta_c``. The solver tries to reduce ``\mu`` by this factor in this iteration.

An example of the output of the example used to explain the interface is
```
iter  time(s)           μ       P-obj       D-obj        gap    P-error    p-error    d-error        α_p        α_d       beta
   1     12.8   1.000e+20   0.000e+00   0.000e+00   0.00e+00   1.00e+10   1.00e+00   6.43e+09   7.51e-01   7.58e-01   3.00e-01
   2     13.2   3.547e+19   3.624e+10  -9.588e+08   1.05e+00   2.49e+09   2.49e-01   1.56e+09   6.97e-01   7.05e-01   3.00e-01
...
  60     13.8   1.216e-14  -2.113e+00  -2.113e+00   2.88e-14   4.53e-76   1.71e-75   8.94e-71   1.00e+00   1.00e+00   1.00e-01
  61     13.8   1.216e-15  -2.113e+00  -2.113e+00   2.88e-15   5.53e-76   7.65e-75   1.67e-70   1.00e+00   1.00e+00   1.00e-01
Optimal solution found
13.806709 seconds (31.85 M allocations: 1.724 GiB, 8.04% gc time, 95.01% compilation time)
iter  time(s)           μ       P-obj       D-obj        gap    P-error    p-error    d-error        α_p        α_d       beta

Primal objective:[-2.1129138814236035493303617046125433190930634590630701102555586784046703753 +/- 8.99e-74]
Primal objective:[-2.1129138814236047658789644991030406911852260781814341016229113808001230237954 +/- 3.25e-77]
Duality gap:[2.878840953931412734986211198407702873449111279309426433372e-16 +/- 2.66e-74]
```
Note that the first iteration takes long because the functions used by the solver get compiled.

### Status
When the algorithm finishes due to one of the termination criteria, the status, the final solution together with the objectives, the used time and an error code is returned. The status can be one of
  - `Optimal`
  - `NearOptimal` -   The solution is primal and dual feasible, and the duality gap is small (``<10^{-8}``), although not smaller than `duality_gap_threshold`.
  - `Feasible`
  - `PrimalFeasible` or `DualFeasible`
  - `NotConverged`

The final solution is stored in the `CLRSResults` structure.
This includes the variables ``(x,X,y,Y)``, the primal and dual objective, and the self-chosen names corresponding to the variables ``Y`` and ``y``.

### Errors
Although unwanted, errors can be part of the output as well. The error codes give an indication what a possible solution could be to avoid the errors.
  0. No error
  1. An arbitrary error. This can be an internal error such as a decomposition that was unsuccessful. If this occurs in the first iteration, it is a strong indication that the constraints are linearly dependent, e.g. due to using a set of sample points which is not minimal unisolvent for the basis used. Otherwise increasing the precision may help. This also includes errors which are due to external factors such as a keyboard interrupt.
  2. The maximum number of iterations has been exceeded. Reasons include: slow convergence, a difficult problem. Possible solutions: increase the maximum number of iterations, increase `gamma` (if `gamma` is small), change the starting solution (`omega_p` and `omega_d`).
  3. The maximum complementary gap (``\mu``) has been exceeded. Usually this indicates (primal and/or dual) infeasibility.
  4. The step length is below the step length threshold. This indicates precision errors or a difficult problem. This may be solved by increasing the initial solutions (`omega_p` and `omega_d`), or by decreasing the step length reduction `gamma`, or by increasing the precision `prec`.

## Multithreading
The solver supports multithreading. This can be used by starting `julia` with
```
julia -t n
```
where `n` denotes the number of threads.
!!! warning
    On Windows, using multiple threads can lead to errors when using multiple clusters and free variables. This is probably related to Arb or the Julia interface to Arb.
