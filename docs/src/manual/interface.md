# Interface

In this section we will explain the interface, which is focused on semidefinite programs with polynomial constraints. The most general form of such a problem is
```math
\begin{aligned}
    \min \quad & \sum_j \langle Y^j, C^j \rangle + \langle y, b\rangle \\
    \text{s.t.} \quad & \langle Y^j, A^j_*(x) \rangle + B^T(x) y = c(x) \\
    & Y^j \succeq 0,
\end{aligned}
```
where we optimize over free scalar variables ``y`` and positive semidefinite variables ``Y^j = \mathrm{diag}(Y^{j,1}, \ldots, Y^{j,L_j})``. The polynomial matrices ``A^j(x)`` are required to be symmetric and have a low-rank block form. That is, ``A^j(x)`` is block-diagonal with blocks ``A^{j,l}(x)`` and
```math
    A^{j,l}(x) = \sum_{r,s=1}^{N_{j,l}} A^{j,l}_{r,s}(x) \otimes E_{r,s}^{R_{j}(l)}
```
where ``E_{r,s}^N`` is the ``N \times N`` matrix with a one in the ``(r,s)`` entry and zeros elsewhere. Furthermore, ``A^{j,l}_{r,s}(x)`` is of low rank such that ``A^{j,l}_{r,s} = (A^{j,l}_{s,r})^T``. Often, the non-diagonal block structure is not used (``N=1`` for all ``j,l``). One example where it is used is in polynomial matrix programs (see, e.g., [de-laat-clustered-2022](@cite)).

This is then converted to a clustered low-rank semidefinite program by [Sampling](@ref).

We will explain the interface using an example from polynomial optimization. Suppose we have some polynomial, e.g.,
```math
f(x,y,z) = x^4 + y^4 + z^4 - 4xyz + x + y + z
```
and we want to find the minimum of this polynomial, or an upper bound on the minimum. We can relax the problem using a sum-of-squares characterization:
```math
\begin{aligned}
    \min \quad & M & \\
    \text{s.t.} \quad & f-M & \text{ is a sum-of-squares}. \\
\end{aligned}
```
Given a vector ``w(x,y,z)`` of basis polynomials up to degree ``d``, we can parametrize a sum-of-squares polynomial ``s`` of degree ``2d`` by a positive semidefinite matrix ``Y`` with
```math
    s = ⟨ Y, ww^{\sf T} ⟩.
```
See the examples section for more complicated examples, explaining more intricate parts of the interface.

```julia
using ClusteredLowRankSolver, AbstractAlgebra, BasesAndSamples

function min_f(d)
    # Set up the polynomial space and define f
    R, (x,y,z) = PolynomialRing(RealField, ["x", "y", "z"])
    f = x^4 + y^4 + z^4 - 4x*y*z + x + y + z

    # Define the objective
    obj = Objective(0, Dict(), Dict(:M => 1))

    # Define the constraint SOS + M = f
    # free variables
    free_dict = Dict(:M => 1)

    # PSD variables (the sum of squares polynomial) & samples
    psd_dict = Dict()
    w = basis_monomial(2d, x, y, z)
    samples = sample_points_simplex(3,2d)
    basis, samples = approximatefekete(w, samples)

    psd_dict[Block(:Y)] = LowRankMatPol([1], [basis[1:binomial(3+d,d)]])

    # the constraint
    con  = Constraint(f, psd_dict, free_dict, samples)

    # Define the polynomial program (maximize obj s.t. con)
    pol_prob = LowRankPolProblem(true, obj, [con])

    # Convert to a clustered low-rank SDP
    sdp = ClusteredLowRankSDP(pol_prob)

    #solve the SDP
    status, sol, time, errorcode = solvesdp(sdp)
end
```

## Objective
For the [`Objective`](@ref), we need the constant, a dictionary with the ``C^{j,l}`` matrices, and a dictionary with the entries of ``b``.
In this case, the constant is 0, the matrix ``C`` is the zero matrix, which can be omitted, and the vector ``b`` has one entry corresponding to the free variable ``M``:
```julia
    # Define the objective
    obj = Objective(0, Dict(), Dict(:M => 1))
```

## Constraints
For a constraint, we need the low-rank matrices ``A^{j,l}_{r,s}(x)``, the entries of ``B^{\sf T}(x)`` and the polynomial ``c(x)``. Furthermore, we require a unisolvent set of sample points.
In our case, ``c = f``, the entry of ``B`` corresponding to ``M`` is the constant polynomial ``1``, and the low-rank matrix corresponding to our PSD matrix variables is the matrix ``ww^{\sf T}``.
With
```julia
    w = monomial_basis(2d, x, y, z)
```
we use `BasesAndSamples` to create the monomial basis for `w`. Then we create a unisolvent set of sample points for polynomials in three variables up to degree ``2d`` by
```julia
    samples = sample_points_simplex(3,2d)
```
In general, this is not a very good combination. We can improve the basis with [`approximatefekete`](@ref), which orthogonalizes the basis with respect to the sample points. If `samples` would contain more samples than needed, this would also select a good subset of the samples.
```julia
    basis, samples = approximatefekete(w, samples)
```
returns the basis in evaluated form, which we can only evaluate on samples from `samples`. Common operations such as multiplications and additions work with these sampled polynomials, but for example extracting the degree is not possible since that requires expressing the polynomials in a graded basis. However, if `w` is a basis ordered on degree, `basis` will have the same ordering.
To create the [`LowRankMatPol`](@ref) for the sum-of-squares polynomial, we use
```julia
    psd_dict[Block(:Y)] = LowRankMatPol([R(1)], [basis[1:binomial(3+d,d)]])
```
Using `Block(:Y)` as key indicates that we use the ``(r,s) = (1,1)`` block of the matrix `:Y`; `Block(:Y, r, s)` would use the `(r,s)` subblock. The size of the matrices is implicitely given by the size of the [`LowRankMatPol`](@ref), which is defined by the prefactors and the rank one terms. In this case, we want to use the basis up to degree `d` to get a sum-of-squares polynomial up to degree `2d`.

The command
```julia
    con  = Constraint(f, psd_dict, free_dict, samples)
```
creates the constraint.

## Low rank polynomial programs
With
```julia
    pol_prob = LowRankPolProblem(true, obj, [con])
```
we create a polynomial problem ([`LowRankPolProblem`](@ref)). The first argument indicates whether we maximize (`true`) or minimize (`false`) the objective (`obj`) with respect to the constraints (in this case one constraint `con`).

## Clustered low rank semidefinite programs
Finally, we convert the polynomial program to a clustered low-rank semidefinite program by sampling with
```julia
    sdp = ClusteredLowRankSDP(pol_prob)
```
which we can solve with
```julia
    status, sol, time, errorcode = solvesdp(sdp)
```
This function has multiple options; see the section with solver [Options](@ref).

## Retrieving variables from the solution
The solver outputs the solution `sol` in the form of a [`CLRSResults`](@ref) struct, which may be used to retrieve variables and the objective from the solver. In the following we want to obtain the (dual) objective of a solution, and the value of a free variable named `:a`.
```julia
    objective = sol.dual_objective
    a_value = sol.freevar[:a]
```
Similarly, we can retrieve the matrix variable `:Y` with `sol.matrixvar[:Y]`.

