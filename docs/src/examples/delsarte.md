# [Example: the Delsarte bound](@id exdelsarte)

In this example we show how the Delsarte linear programming bound [delsarte-spherical-1977](@cite) for the spherical code problem can be modeled and solved using `ClusteredLowRankSolver`. See the [Examples](https://github.com/nanleij/ClusteredLowRankSolver.jl/tree/main/examples) folder for the file with the code. Let ``P_k^n`` be the Gegenbauer polynomial of degree ``k`` with parameter ``n/2-1``, normalized such that ``P_k^n(1) = 1``. The Delsarte bound for dimension ``n``, degree ``2d``, and angle ``\theta`` can be written as
```math
\begin{aligned}
    \min \quad & M && \\
    \text{s.t.} \quad&\sum_{k=1}^{2d} a_k P_k^n(x) \leq -1 && \quad x \in [-1,\cos(\theta)]\\
     & \sum_{k=1}^{2d} a_k  -  M \leq -1 &&\\
     &a_k \geq 0,&&\\
\end{aligned}
```
This gives an upper bound on the size of a spherical code on ``S^{n-1}`` with ``x \cdot y\leq\cos(\theta)`` for any distinct ``x`` and ``y``. Note that we could remove the free variable ``M`` by using the objective ``1+\sum_k a_k``, but for illustration purposes we include this free variable in the formulation.

To model this as a clustered low-rank semidefinite program, we equate the polynomial
```math
- 1 - \sum_k a_k P^n_k(x)
```
to the weighted sums-of-squares polynomial
```math
\langle Y_1,b(x)b(x)^T \rangle + (x+1)(\cos(\theta)-x) \langle Y_2,b(x)b(x)^T\rangle
```
for a vector of basis polynomials ``b(x)``. By a theorem of Lukács, one can always write the polynomial in this form. Moreover, we can truncate the vectors ``b(x)`` such that both terms have degree at most ``2d``. Then we sample the polynomials on ``2d+1`` points ``x_1,\ldots,x_{2d+1}`` in ``[-1,\cos(\theta)]`` to obtain semidefinite constraints (any set of ``2d+1`` distinct points is minimal unisolvent for the univariate polynomials up to degree ``2d``).

Furthermore, to model the linear inequality constraint, we introduce a slack variable ``s \geq 0`` such that we need ``\sum_{k=1}^{2d} a_k + s - M = -1``.
Together this gives the semidefinite program
```math
\begin{aligned}
    \min \quad & M && \\
    \text{s.t.} &\sum_{k=1}^{2d} a_k P_k^n(x_l) + \langle b_d(x_l)b_d(x_l)^T, Y_1 \rangle &&\\
    &\quad + \langle g(x_l)b_{d-1}(x_l)b_{d-1}(x_l)^T, Y_2 \rangle &= -1,\quad& l=1, \ldots, 2d+1 \\
     & \sum_{k=1}^{2d} a_k + s - M & =-1 \quad &\\
     &a_k \geq 0&&\\
     &s \geq 0&& \\
     &Y_i \succeq 0, &&i=1,2,
\end{aligned}
```
where ``b_d`` is a basis of polynomials up to degree ``d`` and
```math
g(x) = (x+1)(\cos(\theta)-x).
```

Now that we formulated the semidefinite program, we can model it in Julia.
We use the `polynomial_ring` of `AbstractAlgebra` to define the polynomials.
We will start with the objective and constraints after which we define the complete program.

The objective is simple, since we only optimize the value of the free variable ``M``. This gives the start of our script:
```@example 1; continued = true
using ClusteredLowRankSolver, AbstractAlgebra

function delsarte(n, d, costheta)

    # Initialize the objective with additive constant 0,
    # no dependence on matrix variables,
    # and dependence with coefficient 1 on the free variable :M
    obj = Objective(0, Dict(), Dict(:M => 1))

```
For the first constraint, we need the Gegenbauer polynomials, which are part of `ClusteredLowRankSolver`. In this constraint we only use the positive semidefinite matrix variables, for which we collect the constraint matrices in a `Dict`:
```@example 1; continued = true
    R, x = polynomial_ring(RealField, :x)
    # A vector consisting of the Gegenbauer polynomials
    # of degree at most 2d
    gp = basis_gegenbauer(2d, n, x)
    psd_dict1 = Dict()
    for k = 1:2d
        psd_dict1[(:a, k)] = [gp[k+1];;]
    end
```
Note that here we use a general (high-rank) constraint matrix for the variables ``a_k``.
For the sums-of-squares part, we define a basis and the set of samples points. We use `approximatefekete` to obtain a basis which is orthogonal with respect to the sample points.
```@example 1; continued = true
    # 2d+1 Chebyshev points in the interval [-1, cos(θ)]
    samples = sample_points_chebyshev(2d, -1, costheta)
    # A basis for the Chebyshev polynomials of degree at most 2d
    basis = basis_chebyshev(2d, x)
    # Optimize the basis with respect to the samples
    sosbasis, samples = approximatefekete(basis, samples)
```
Since the second sum-of-squares has a weight of degree 2, we use a basis up to degree ``d-1`` in that case.
```@example 1; continued = true
    psd_dict1[(:SOS, 1)] = LowRankMatPol([1], [sosbasis[1:d+1]])
    psd_dict1[(:SOS, 2)] = LowRankMatPol([(1+x)*(costheta-x)], [sosbasis[1:d]])
    constr1 = Constraint(-1, psd_dict1, Dict(), samples)
```
In the last line, we define the constraint (with constant polynomial ``-1``). The extra dictionary corresponds to the coefficients for the free variables, which we do not use in this constraint.

The second constraint looks simpler since it does not involve polynomials. We also introduce a slack variable, this gives the constraint ``\sum_k a_k + s - M = -1``, where ``s \geq 0``.
```@example 1; continued = true
    # The free variable M has coefficient -1:
    free_dict2 = Dict(:M => -1)
    # The variables a_k and the slack variable have coefficient 1:
    psd_dict2 = Dict()
    for k = 1:2d
        psd_dict2[(:a, k)] = [1;;]
    end
    psd_dict2[:slack] = [1;;]
    constr2 = Constraint(-1, psd_dict2, free_dict2)
```
Note that we again use a general matrix for the variables ``a_k``. The use of low-rank matrices and general matrices should be consistent per variable.
Now we can define the `Problem`.
```@example 1; continued = true
    problem = Problem(Minimize(obj), [constr1, constr2])
```
and solve the corresponding semidefinite program with
```@example 1; continued=true
    status, dualsol, primalsol, time, errorcode = solvesdp(problem)
```
after which we return the objective value.
```@example 1
    return objvalue(problem, primalsol)
end
nothing # hide
```
Alternatively, we can first create the semidefinite program with
```julia
    sdp = ClusteredLowRankSDP(problem)
```
after which we can call `solvesdp` on `sdp`.

Running this for one of the known sharp cases gives
```@example 1
delsarte(8, 3, 1//2)
```