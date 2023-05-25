# Example: the Delsarte bound

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
We use the `PolynomialRing` structure of `AbstractAlgebra` to define the polynomials. For the bases and samples, we use the auxilliary package `BasesAndSamples`. And of course, we also need `ClusteredLowRankSolver`.
We will start with the objective and constraints after which we define the complete program.

The objective is simple, since we only optimize the value of the free variable ``M``. This gives the start of our script:
```julia
using ClusteredLowRankSolver, BasesAndSamples, AbstractAlgebra

function delsarte(n=3, d=10, costheta=1//2)

    # Initialize the objective with additive constant 0,
    # no dependence on matrix variables,
    # and dependence with coefficient 1 on the free variable :M
    obj = Objective(0, Dict(), Dict(:M => 1))

```
For the first constraint, we need the Gegenbauer polynomials, which are  defined in the package `BasesAndSamples`. In this constraint we only use the positive semidefinite matrix variables, for which we collect the constraint matrices in a `Dict`:
```julia
    R, (x,) = PolynomialRing(RealField, ["x"])
    # A vector consisting of the Gegenbauer polynomials
    # of degree at most 2d
    gp = basis_gegenbauer(2d, n, x)
    psd_dict1 = Dict()
    for k = 1:2d
        # The eigenvalues are the gegenbauer polynomials,
        # and the vectors have the polynomial 1 as entry:
        psd_dict1[Block((:a, k))] = hcat([gp[k+1]])
    end
```
Since Julia 1.7, it is possible to construct matrices with ``[gp[k+1];;]`` instead of ``hcat``. Note that here we use a general (high-rank) constraint matrix for the variables ``a_k``.
For the sums-of-squares part, we define a basis and the set of samples points. We use `approximatefekete` to obtain a basis which is orthogonal on the sample points.
```julia
    # 2d+1 Chebyshev points in the interval [-1, cos(θ)]
    samples = sample_points_chebyshev(2d, -1, costheta)
    # A basis for the Chebyshev polynomials of degree at most 2d
    basis = basis_chebyshev(2d, x)
    # Optimize the basis with respect to the samples
    sosbasis, samples = approximatefekete(basis, samples)
```
The samples need to be a vector of numbers, because we also can consider multivariate polynomials.
Since the second sum-of-squares has a weight of degree 2, we use a basis up to degree ``d-1`` in that case.
```julia
    psd_dict1[Block((:SOS, 1))] =
        LowRankMatPol([1], [sosbasis[1:d+1]])
    psd_dict1[Block((:SOS, 2))] =
        LowRankMatPol([(1+x)*(costheta-x)], [sosbasis[1:d]])
    constr1 = Constraint(-1, psd_dict1, Dict(), samples)
```
In the last line, we define the constraint (with constant polynomial ``-1``). The extra dictionary corresponds to the coefficients for the free variables, which we do not use in this constraint.

The second constraint seems simpler since it does not involve polynomials. However, since we first define a `LowRankPolProblem`, we need to write it as a constraint with polynomials of degree 0. We also introduce a slack variable, this gives the constraint ``\sum_k a_k + s - M = -1``, where ``s \geq 0``.
```julia
    # The free variable M has coefficient -1:
    free_dict2 = Dict(:M => -1)
    # The variables a_k and the slack variable have coefficient 1:
    psd_dict2 = Dict()
    for k = 1:2d
        psd_dict2[Block((:a, k))] = hcat([1])
    end
    psd_dict2[Block(:slack)] = LowRankMatPol([1], [[1]])
    # We have one sample, which is arbitrary
    constr2 = Constraint(-1, psd_dict2, free_dict2, [[0]])
```
Note that we again use a general matrix for the variables ``a_k``. The use of low-rank matrices and general matrices should be consistent per variable.
Now we can define the `LowRankPolProblem` and convert it to a `ClusteredLowRankSDP`. When converting, we can choose to model some of the PSD variables as free variables, which can be beneficial if the corresponding constraint matrices are of relatively high rank or if they couple a lot of polynomial constraints. In this case, we just illustrate the option by modelling ``a_0`` as free variable. This adds the constraint ``X = a_0`` with a positive semidefinite variable ``X`` (in this case a ``1 \times 1`` matrix), and replaces ``a_0`` by a free variable in every constraint.
```julia
    # The first argument is false because we minimize:
    sos = LowRankPolProblem(false, obj, [constr1, constr2])
    sdp = ClusteredLowRankSDP(sos; as_free = [(:a, 0)])
```
Now we can solve the semidefinite program with
```julia
    status, result, time, errorcode = solvesdp(sdp)
end
```
