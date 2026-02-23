# [The solver](@id solver)
[`ClusteredLowRankSolver.jl`](https://github.com/nanleij/ClusteredLowRankSolver.jl) implements a primal-dual interior-point method. That is, it solves both the primal and the dual problem. The problem given to the solver is considered to be in dual form. For more information on the primal-dual algorithm, see [de-laat-clustered-2022](@cite) and [simmons-duffin-semidefinite-2015](@cite).

A [`Problem`](@ref) can be solved using the function [`solvesdp`](@ref). This first converts the problem to a [`ClusteredLowRankSDP`](@ref), after which it is solved using the algorithm.
```@docs
solvesdp
``` 

## Options
Here we list the most important options. For the remaining options, see the documentation and the explanation of the algorithm in [simmons-duffin-semidefinite-2015](@cite).
  -  `prec` - The number of bits used for the calculations. The default is the `BigFloat` precision, which defaults to 256 bits.
  - `duality_gap_threshold` - Gives an indication of how close the solution is to the optimal solution. As a rule of thumb, a duality gap of ``10^{-(k+1)}`` gives ``k`` correct digits.  Default: ``10^{-15}``
  - `gamma` - The step length reduction; if a step of ``\alpha`` is possible, a step of ``\min(\gamma \alpha, 1)`` is taken. A lower `gamma` results in a more stable convergence, but can be significantly slower. Default: ``0.9``.
  - `omega_p`, `omega_d` - The size of the initial primal respectively dual solution. A low `omega` can keep the solver from converging, but a high `omega` in general increases the number of iterations needed and thus also the solving time. Default: ``10^{10}``
  - `need_primal_feasible`, `need_dual_feasible` - If `true`, terminate when a primal or dual feasible solution is found, respectively. Default: `false`.
  - `primal_error_threshold`, `dual_error_threshold` - The threshold below which the primal and dual error should be to be considered primal and dual feasible, respectively. Default: ``10^{-15}``.


## Output
When the option `verbose` is `true` (default), the solver will output information for every iteration.
In order of output, we have (where the values are from the start of the iteration except for the step lengths, which are only known at the end of the iteration)
  - The iteration number
  - The time since the start of the first iteration
  - The complementary gap ``\mu = \langle X, Y \rangle / K`` where ``K`` is the number of rows of ``X``. Here ``X`` and ``Y`` denote the primal and dual solution matrices. The solution will converge to the optimum for ``\mu \to 0``.
  - The dual objective
  - The primal objective
  - The relative duality gap 
  - The dual matrix error 
  - The dual scalar error 
  - The primal (scalar) error 
  - The dual step length
  - The primal step length
  - ``\beta_c``. The solver tries to reduce ``\mu`` by this factor in this iteration.

An example of the output of the [Example](@ref expolyopt) from polynomial optimization is
```
iter  time(s)           μ       P-obj       D-obj        gap    P-error    p-error    d-error        α_p        α_d       beta
    1     11.9   1.000e+20   0.000e+00   0.000e+00   0.00e+00   1.00e+10   1.00e+00   1.95e+10   7.42e-01   7.10e-01   3.00e-01
    2     13.4   3.995e+19   1.999e+11  -2.907e+09   1.03e+00   2.58e+09   2.58e-01   5.65e+09   7.46e-01   7.17e-01   3.00e-01
    3     13.4   1.576e+19   3.079e+11  -4.779e+09   1.03e+00   6.53e+08   6.53e-02   1.60e+09   7.32e-01   7.31e-01   3.00e-01
...
   55     13.9   5.066e-14  -2.113e+00  -2.113e+00   8.39e-14   8.64e-78   2.59e-77   8.21e-73   1.00e+00   1.00e+00   1.00e-01
   56     13.9   5.067e-15  -2.113e+00  -2.113e+00   8.39e-15   8.64e-78   8.64e-78   8.39e-73   1.00e+00   1.00e+00   1.00e-01
Optimal solution found
 13.860834 seconds (13.74 M allocations: 913.073 MiB, 7.72% gc time, 93.09% compilation time)
 iter  time(s)           μ       P-obj       D-obj        gap    P-error    p-error    d-error        α_p        α_d       beta

Dual objective:-2.112913881423601867325289796075301826150007716044362101360781221096092533872562
Primal objective:-2.112913881423605414349991239275382883067580432169230529548206052006356176913883
Duality gap:8.393680245626824434313082297089851809408852609517159688543365552836941907249006e-16
```
Note that the first iteration takes long because the functions used by the solver get compiled.
The function [`solvesdp`](@ref) returns the status of the solutions, the dual and primal solutions, the solve time and an error code (see below).

### Status
When the algorithm finishes due to one of the termination criteria, the status, the final solution together with the objectives, the used time and an error code is returned. The status can be one of
  - `Optimal`
  - `NearOptimal` -  The solution is primal and dual feasible, and the duality gap is small (``<10^{-8}``), although not smaller than `duality_gap_threshold`.
  - `Feasible`
  - `PrimalFeasible` or `DualFeasible`
  - `NotConverged`

### Errors
Although unwanted, errors can be part of the output as well. The error codes give an indication what a possible solution could be to avoid the errors.
  0. No error
  1. An arbitrary error. This can be an internal error such as a decomposition that was unsuccessful. If this occurs in the first iteration, it is a strong indication that the constraints are linearly dependent, e.g. due to using a set of sample points which is not minimal unisolvent for the basis used. Otherwise increasing the precision may help. This also includes errors which are due to external factors such as a keyboard interrupt.
  2. The maximum number of iterations has been exceeded. Reasons include: slow convergence, a difficult problem. Possible solutions: increase the maximum number of iterations, increase `gamma` (if `gamma` is small), change the starting solution (`omega_p` and `omega_d`).
  3. The maximum complementary gap (``\mu``) has been exceeded. Usually this indicates (primal and/or dual) infeasibility.
  4. The step length is below the step length threshold. This indicates precision errors or a difficult problem. This may be solved by increasing the initial solutions (`omega_p` and `omega_d`), or by decreasing the step length reduction `gamma`, or by increasing the precision `prec`. If additionally the complementary gap is increasing, it might indicate (primal and/or dual) infeasibility.

## Multithreading
The solver supports multithreading. This can be used by starting `julia` with
```
julia -t n
```
where `n` denotes the number of threads.
!!! warning
    On Windows, using multiple threads can lead to errors when using multiple clusters and free variables. This is probably related to Arb or the Julia interface to Arb.
