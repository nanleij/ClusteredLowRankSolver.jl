# [Example: Clustering](@id exClustering)
Here we give a small example in which it is beneficial to use the option `as_free` to model positive semidefinite variables as free variables. We focus on the constraints. Suppose you want to solve a semidefinite program with the constraint that 
```math
\begin{align*}
    \langle Y, A(x) \rangle
\end{align*}
```
is nonnegative on a union of ``k`` semialgebraic sets 
```math
\begin{align*}
    G_i = \{x \in \mathbb{R}^n : g_{i,j}(x) \geq 0, j=1, \ldots, m_i\}
\end{align*}
```
and suppose that these semialgebraic sets are archimedean, so that we can use Putinar's theorem. Then this translates into ``k`` sum-of-squares constraints; one for each semialgebraic set.

Assuming that the matrix `A` is defined before, as well as the polynomials `g[i][j]`, the basis `sosbasis[i][j]` of the correct degrees and the sample points `samples`, this gives the code
```julia
    constraints = []
    for i=1:k
        psd_dict = Dict()
        # this is the same everywhere
        psd_dict[:Y] = A
        for j=1:m[i]
            # this differs per constraint
            # note that different sum-of-square matrices have different names
            psd_dict[(:sos,i,j)] = LowRankMatPol([-g[i][j]], [sosbasis[i][j]])
        end
        push!(constraints, Constraint(0,psd_dict,Dict(), samples))
    end
```
Since the positive semidefinite matrix variable ``Y`` occurs in every constraint, the corresponding cluster contains ``k \cdot |S|`` constraints after sampling, where ``|S|`` is the number of samples. To split this into ``k`` clusters of ``|S|`` constraints, we use the function  [`model_psd_variables_as_free_variables`](@ref) to model ``Y`` as free variables:
```julia
    problem = Problem(Minimize(obj), constraints)
    problem = model_psd_variables_as_free_variables(problem, [:Y])
```
This adds auxilliary free variables ``X_{ij}``, adds the constraints ``X_{ij} = Y_{ij}``, and replaces the ``Y_{ij}`` in the constraints by ``X_{ij}``. Then the only positive semidefinite variables in the polynomial constraints are the sums-of-squares matrices, which causes each sums-of-squares constraint to be assigned to its own cluster.

