# [Example: Rounding the Delsarte LP bound](@id exrounding)

For this example, we slightly modify the code of the [Delsarte](@ref exdelsarte) bound to define an exact problem.
```@example rounding
using ClusteredLowRankSolver, Nemo

function delsarte_exact(n, d, costheta; FF=QQ, g=1, eps=1e-40)
    constraints = []

    P, x = polynomial_ring(FF, :x)

    gbasis = basis_gegenbauer(2d, n, x)
    sosbasis = basis_chebyshev(2d, x)

    samples = sample_points_chebyshev(2d)
    # round the samples to QQ:
    samples = [round(BigInt, x * 10^4)//10^4 for x in samples]

    c = Dict()
    for k = 0:2d
        c[k] = [gbasis[k+1];;]
    end
    c[:A] = LowRankMatPol([1], [sosbasis[1:d+1]])
    c[:B] = LowRankMatPol([(x+1)*(costheta-x)], [sosbasis[1:d]])
    push!(constraints, Constraint(-1, c, Dict(), samples))

    objective = Objective(1, Dict(k => [1;;] for k=0:2d), Dict())

    problem = Problem(Minimize(objective), constraints)

    problem_bigfloat = map(x->generic_embedding(x, g), problem)
    status, primalsol, dualsol, time, errorcode = solvesdp(problem_bigfloat, duality_gap_threshold=eps)

    return objvalue(problem, dualsol), problem, primalsol, dualsol
end
nothing #hide
```
Here we made a few modifications. 
  - We defined the polynomial ring over the field `FF`, so that the problem becomes exact.
  - We also made the samples exact rational numbers. This is in this case not strictly necessary if we use coefficient matching for the rounding procedure, but it is necessary when evaluating polynomials on these samples.
  - We added the line
    ```julia
        problem_bigfloat = map(x->generic_embedding(x, g), problem)
    ```
    This embeds the field in ``\mathbb{R}`` using the floating point approximation `g` of the generator of the field we use. The exact problem cannot be solved with the solver if we use a number field with a generator, since it is unclear which generator of the field we meant when building the problem.
  - We now return the problem and the primal and dual solution too. 

One example where the delsarte bound is sharp, is when considering a code with ``\cos\theta = 1/(\sqrt{5} - 1)`` in ``\mathbb{R}^4``. The optimal spherical code then has ``120`` points. To round the solution, we first define the field using the minimal polynomial ``x^2 - 5 = 0``.
```@example rounding
R, x = polynomial_ring(QQ, :x)
N, z = number_field(x^2 - 5, :z)
gapprox = sqrt(big(5))
nothing # hide
```
Then we can call the rounding procedure:
```@example rounding
n = 4
d = 9
costheta = 1/(z-1)
# find an approximate solution
obj, problem, primalsol, dualsol = delsarte_exact(n, d, costheta; FF=N, g = gapprox)
# round the approximate solution to an exact solution
success, exactdualsol = exact_solution(problem, primalsol, dualsol; FF=N, g=gapprox)
objexact = objvalue(problem, exactdualsol)
(success, objexact)
```
When the problem is not defined over the same field as the solution, we can find the field using
```@example rounding
N2, gapprox2 = find_field(primalsol, dualsol)
defining_polynomial(N2), gapprox2
```
which returns the field with defining polynomial ``x^2 + x - 1 = 0`` and approximate generator ``-1.618033... \approx (-1 -\sqrt{5})/2``.

## Using coefficient matching
As mentioned in the section about [Rounding](@ref secrounding), using coefficient matching will often result in a solution of smaller bit size. To do that in this example, one can use
```@example rounding
obj, problem, primalsol, dualsol = delsarte_exact(n, d, costheta; FF=N, g = gapprox)
R, x = polynomial_ring(N, :x)
mon_basis = [x^k for k=0:2d]
success, exactdualsol_smaller = exact_solution(problem, primalsol, dualsol, FF=N, g=gapprox, monomial_bases = [mon_basis])
nothing # show the full output, this time? # hide
```
This is recommended especially for larger polynomial programs.

## Getting an exact feasible solution close to the optimum

To modify the code to find an exact strictly feasible solution, we change the code for the objective to the following:
```julia
   if isnothing(obj)
      objective = Objective(1, Dict(k => [1;;] for k=0:2d), Dict())
   else
      objective = Objective(0, Dict(), Dict()) 
      push!(constraints, Constraint(obj-1, Dict(k => [1;;] for k=0:2d), Dict()))
   end
```
where the function signature is now given by
```julia
function delsarte_exact(n, d, costheta; obj=nothing, FF=QQ, g=1, eps=1e-40)
```
That is, it takes an extra argument `obj`. When `obj` is set to a numerical value, this adds the constraint that the objective should be equal to `obj`. Then the exact solution can be found as follows.
```@setup rounding2
using ClusteredLowRankSolver, Nemo

function delsarte_exact(n, d, costheta; obj=nothing, FF=QQ, g=1, eps=1e-40)
    constraints = []

    P, x = polynomial_ring(FF, :x)

    gbasis = basis_gegenbauer(2d, n, x)
    sosbasis = basis_chebyshev(2d, x)

    samples = sample_points_chebyshev(2d)
    # round the samples to QQ:
    samples = [round(BigInt, x * 10^4)//10^4 for x in samples]

    c = Dict()
    for k = 0:2d
        c[k] = [gbasis[k+1];;]
    end
    c[:A] = LowRankMatPol([1], [sosbasis[1:d+1]])
    c[:B] = LowRankMatPol([(x+1)*(costheta-x)], [sosbasis[1:d]])
    push!(constraints, Constraint(-1, c, Dict(), samples))

    if isnothing(obj)
        objective = Objective(1, Dict(k => [1;;] for k=0:2d), Dict())
    else
        objective = Objective(0, Dict(), Dict()) 
        push!(constraints, Constraint(obj-1, Dict(k => [1;;] for k=0:2d), Dict()))
    end

    problem = Problem(Minimize(objective), constraints)

    problem_bigfloat = map(x->generic_embedding(x, g), problem)
    status, primalsol, dualsol, time, errorcode = solvesdp(problem_bigfloat, duality_gap_threshold=eps)

    return objvalue(problem, dualsol), problem, primalsol, dualsol
end
```
```@example rounding2
d = 10
# find the objective
obj_initial, problem_initial, _, _ = delsarte_exact(3, d, 1//2)
# find a strictly feasible solution with a slightly larger objective
obj = obj_initial + 1e-6
_, problem, primalsol, dualsol = delsarte_exact(3, d, 1//2, obj=rationalize(obj))
# round the solution
successfull, exactdualsol = exact_solution(problem, primalsol, dualsol)
Float64(objvalue(problem_initial, exactdualsol)), obj_initial
```
