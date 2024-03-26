# Tutorial

In this tutorial you will learn
  - to build a [`Problem`](@ref), which includes
    - creating the objective
    - creating a constraint
    - combining them into a [`Problem`](@ref)
  - to solve a [`Problem`](@ref)
  - to find an exact optimal solution

For explanation on more intricate parts of the interface, see [Advanced modeling](@ref); for additional examples see the Examples section.


## Running example

We will explain the interface using an example from polynomial optimization. Suppose we want to find the global minimum (or a lower bound on this) of the polynomial
```math
f(u,v,t,w) = -ut^3+4vt^2w + 4 utw^2 + 2vw^3 + 4  ut+ 4t^2 -10 vw - 10  w^2 +2
```
in the domain ``[-1/2, 1/2]^4``. 
We can relax the problem using a sum-of-squares characterization:
```math
\begin{aligned}
    \text{maximize} \quad & M & \\
    \text{subject to} \quad & f-M & = s_0 + \sum_i w_i s_i. \\
\end{aligned}
```
where the ``w_i`` are polynomial weights describing the domain and ``s_i`` are sum-of-squares polynomials. In this example we use
```math
\begin{aligned}
    [-1/2, 1/2]^4 = \{(u,v,t,w): p(u), p(v), p(t), p(w) \geq 0\}
\end{aligned}
``` 
with ``p(x) = 1/4 - x^2``.
Given a vector ``m`` whose entries form a basis of the space of polynomials up to degree ``d``, we can parametrize a sum-of-squares polynomial ``s`` of degree ``2d`` by a positive semidefinite matrix ``Y`` with
```math
    s = ⟨ Y, mm^{\sf T} ⟩.
```

The problem parameters are now 
  - the polynomial ``f``
  - the weight polynomials ``w_i``
  - the degree of the relaxation ``d``
We will start with building a function that takes these parameters, and constructs the [`Problem`](@ref). Since the user will give the polynomial `f` and the weights, we need to extract the polynomial ring and the polynomial variables.

```@example running; continued=true
using ClusteredLowRankSolver, Nemo
function min_f(f, ws, d; basis_change=true)
    # extract the polynomial ring
    R = parent(f)
    x = gens(R)
    n = nvars(R)
```

## Defining the objective
First we will define the [`Objective`](@ref). This consists of a constant offset, a dictionary with the coefficient matrices for the positive semidefinite matrix variables appearing in the objective, and a dictionary with the scalar coefficients for the free variables used in the objective. In this example, 
  - the constant offset is `0`, 
  - the dictionary of matrix coefficients is empty because we do not use the matrix variables in the objective,
  - and the dictionary of scalar coefficients has one entry corresponding to ``M`` 
This gives the first part of our function:
```@example running; continued=true
    # Define the objective
    obj = Objective(0, Dict(), Dict(:M => 1))
```

```@docs
Objective
```
## Defining the constraints
Now we will define the constraints. In `ClusteredLowRankSolver` a `Constraint` is of the form
```math
    \sum_i \langle A_i, Y_i \rangle + \sum_j b_j y_j = c. 
```
To define it, we need the right-hand side ``c``, the matrix coefficients ``A_i`` for the positive semidefinite matrix variables, and the coefficients ``b_j`` for the free variables. Here the scalars ``c`` and ``b_j`` and the entries of ``A_i`` can either be constants or polynomials. When these are polynomials, we also need a unisolvent set of samples.
In this example, 
  - the right-hand side ``c`` is the polynomial ``f``, 
  - the matrix coefficients ``A_i`` are the weight polynomials times a suitable low rank matrix of the form ``mm^{\sf T}``, 
  - and we have one free variable ``M``, with coefficient ``1``.
#### Defining the polynomial basis and the samples
We will first define the vector ``m`` of basis polynomials, and the samples. For simplicity we will use the monomial basis, and the samples defined by the rational points in the simplex with denominator ``2d``. 
```@example running; continued=true
    basis = basis_monomial(2d, x...)
    samples = sample_points_simplex(n, 2d; T=Rational{BigInt})
```
See [Sampling](@ref) for more explanation on sampling and unisolvence. In more complicated situations, we can improve the basis with [`approximatefekete`](@ref). This orthogonalizes the basis with respect to the sample points, and if `samples` contains more samples than needed, this selects a good subset of the samples. We include this in the function using a keyword argument `basis_change=true` and
```@example running; continued=true
    if basis_change
        basis, samples = approximatefekete(basis, samples)
    end
```
This returns a basis of `SampledMPolyRingElem`'s, which we can only evaluate at samples from `samples`. Common operations such as multiplications and additions work with these sampled polynomials, but operations such as extracting the degree is not possible since that requires expressing the polynomials in a graded basis. However, if the initial `basis` is ordered on degree, the final `basis` will have the same ordering, so we store the degrees using the code
```@example running; continued=true
    degrees = total_degree.(basis_monomial(2d, x...))
```

#### Defining the coefficients for the constraint
Now we are ready to define the low-rank matrix coefficients for the sum of squares polynomials. 
For each weight polynomial, we need to add the matrix coefficient corresponding to that weight to the dictionary of matrix coefficients. Since we want polynomials of degree `2d`, we first need to select the part of the basis that we use in the sum-of-squares polynomials.
```@example running; continued=true
    psd_dict = Dict()
    for (i, w) in enumerate(ws)
        basispart = [basis[j] for j in eachindex(basis) if 2degrees[j] + total_degree(w) <= 2d]
        psd_dict[(:sos, i)] = LowRankMatPol([w], [basispart])
    end
```
The size of the matrices is implicitely given by the size of the [`LowRankMatPol`](@ref), which is defined by the prefactors and the rank one terms.

Similarly, we can create the dictionary with the free variables using 
```@example running; continued=true
    free_dict = Dict(:M => 1)
```
#### Defining the constraint
Now we can construct the [`Constraint`](@ref) with
```@example running; continued=true
    con = Constraint(f, psd_dict, free_dict, samples)
```
From the constraint, the dictionaries and coefficients can be retrieved using [`matrixcoeff`](@ref)(s) and [`freecoeff`](@ref)s.

Specifying non-polynomial constraints works similarly, in which case no samples should be supplied to the [`Constraint`](@ref) struct.
```@docs
Constraint
LowRankMatPol
approximatefekete
matrixcoeff
matrixcoeffs
freecoeff
freecoeffs
```


## Defining the problem
Now that we have the objective and the constraint, we can create the [`Problem`](@ref) with
```@example running; continued=true
    problem = Problem(Maximize(obj), [con])
```
Here the first argument gives the objective and the optimization sense (maximization or minimization). The second argument is a vector of the constraints defining the feasible region (in this example, only the constraint `con`). The objective and constraints can be retrieved from the problem using [`objective`](@ref) and [`constraints`](@ref). Instead of first constructing all constraints and then defining the [`Problem`](@ref), it is also possible to first define the problem using the objective, and then add the constraints with the function [`addconstraint!`](@ref)

```@docs
Problem
Maximize
Minimize
objective
constraints
addconstraint!
```

## Checking for obvious mistakes

Some basic checks on the problem and/or semidefinite program can be done using [`check_problem`](@ref) and [`check_sdp!`](@ref).
The [`check_problem`](@ref) function checks that
  - the sizes of the vectors in the low-rank constraint matrices are the same,
  - all constraints use at least one positive semidefinite matrix variable,
  - all variables in the objective are actually used in the constraints.
The [`check_sdp!`](@ref) function 
  - checks that the constraint matrices are symmetric,
  - removes empty matrices and zero matrices.
```@example running; continued = true
    @assert check_problem(problem)
```
```@docs
check_problem
check_sdp!
```

## Solving the problem

We can solve the [`Problem`](@ref) with the function [`solvesdp`](@ref):
 ```@example running; continued=true
    status, primalsol, dualsol, time, errorcode = solvesdp(problem; prec=512, duality_gap_threshold=1e-60)
```
This function has multiple options including for example the number of bits used for solving the semidefinite program (`prec`) and how close the solution should be to optimality (`duality_gap_threshold`); see the page about the [solver](@ref solver) for more information.

Alternatively, it is possible to explicitely convert the problem into a semidefinite program using
```julia
    sdp = ClusteredLowRankSDP(problem)
```
which can also be solved with
```julia
    status, sol, time, errorcode = solvesdp(sdp)
```
This is for example useful when you wish to save the semidefinite program (e.g., using `Serialization`), or when you want to perform the extra checks provided by [`check_sdp!`](@ref).
```@docs; canonical=false
solvesdp
```
```@docs
ClusteredLowRankSDP
```

## Retrieving variables from the solution
The solver returns, among others, the primal and dual solutions as `PrimalSolution{BigFloat}` and `DualSolution{BigFloat}`. To retrieve the variables, it is possible to use
```julia
    matrixvar(dualsol, (:sos, 1))
```
and `freevar(dualsol, :M)` for free variables. Similarly, use [`matrixvars`](@ref) and [`freevars`](@ref) to iterate over all variables:
```julia
    for (k, m) in matrixvars(dualsol)
        # do stuff with the matrix m or the name k
    end 
```
To compute the objective for this solution, we can use
```@example running; continued=true
    objective = objvalue(problem, dualsol)
```
We might also want to return the problem and the solutions:
```@example running
    return objective, problem, primalsol, dualsol
end
nothing # hide
```
```@docs
PrimalSolution
DualSolution
matrixvar
matrixvars
freevar
freevars
objvalue
```

## Running the example
Now we can define the problem parameters and run the function
```@example running
d = 2
R, (u,v,t,w) = polynomial_ring(QQ, 4)
f = -u*t^3 + 4v*t^2*w + 4u*t*w^2 + 2v*w^3 + 4u*t+ 4t^2 - 10v*w - 10w^2 +2
# the function for the weights
p(x) = 1//4 - x^2
ws = [R(1), p(u), p(v), p(t), p(w)]
# call the function
obj, problem, primalsol, dualsol = min_f(f, ws, d)
obj
```

## Rounding the solution

Here we explain how we can heuristically extract an exact optimal solution from the numerical solution. This approach has been developed in the paper [CLL24](@cite), and we refer to this paper for more information on the assumptions and the method.
See [Rounding](@ref secrounding) for more information on how to use the rounding implementation. 

Since we expect the solution to be relatively nice in terms of polynomials, we avoid doing the basis change generated by `approximatefekete`, and try to find the field using the following code. 
```@example running
obj, problem, primalsol, dualsol = min_f(f, ws, 2; basis_change=false)
N, gapprox = find_field(primalsol, dualsol)
N
```
We find a field of degree 2. Trying to round the objective to this field gives
```@example running
to_field(obj, N, gapprox)
println(ans)# stuff to print it nicely since Nemo doesn't print nicely # hide
```
Since this is a small expression, this gives some indication that `N` is the correct field. Now we will try to round the numerical solution to this field.
First we convert the problem to the field `N`. This can be done using the function [`generic_embedding`](@ref). 
```@example running
problem = map(x->generic_embedding(x, gen(N), base_ring=N), problem)
nothing # hide
```
In general it is advisable to use coefficient matching to define the semidefinite program for the rounding procedure. For this we need to build a monomial basis for each constraint. 
```@example running
R, x = polynomial_ring(N, 4)
monbasis = basis_monomial(4, x...)
nothing # hide
```
Then we are ready to try the rounding procedure. This will not always succeed without tuning the settings, but in this example it does. See the [Rounding](@ref secrounding) page for information on the settings.
```@example running
success, exactdualsol = exact_solution(problem, primalsol, dualsol;
        FF=N, g=gapprox, monomial_bases=[monbasis])
nothing # hide
```
If `success` is `true`, the solution `exactdualsol` is guaranteed to be feasible. The affine constraints have been checked in exact arithmetic, and positive semidefiniteness has been checked in ball arithmetic.  
We can check that we get the same objective as directly rounding the objective to the field `N`.
```@example running
objvalue(problem, exactdualsol)
println(ans)# stuff to print it nicely since Nemo doesn't print nicely # hide
```
The output of the rounding procedure to the terminal contains some information about the number of kernel vectors and the largest numbers in absolute value occuring in those kernel vectors, some progress messages, and some messages about the system that is solved.  

```@docs; canonical=false
find_field
exact_solution
```
```@docs
generic_embedding
to_field
```



