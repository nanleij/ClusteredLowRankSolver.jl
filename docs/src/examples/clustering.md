# Example: Clustering
Here we give a small example in which it is beneficial to use the option ``as_free`` to model positive semidefinite variables as free variables. We focus on the constraints. Suppose you want to solve a semidefinite program with the constraint that 
```math
\begin{align*}
    \langle Y, A(x) \rangle
\end{align*}
```
is nonnegative on a union of ``k`` semialgebraic sets 
```math
\begin{align*}
    G_i = \{x \in \mathbb{R}^n : g_j^i(x) \geq 0, j=1, \ldots, m_i\}
\end{align*}
```
and suppose that these semialgebraic sets are archimedean, so that we can use Putinar's theorem. Then this translates into ``k`` sum-of-squares constraints; one for each semialgebraic set.

Assuming that the low-rank matrix ``A`` is defined before, as well as the polynomials ``g[i][j]``, the basis ``sosbasis[i][j]`` of the correct degrees and the sample points ``samples``, this gives the code
```julia
    constraints = []
    for i=1:k
        psd_dict = Dict()
        psd_dict[Block(:Y)] = A
        for j=1:m[i]
            psd_dict[Block((:sos,i,j))] = LowRankMatPol([-g[i][j]], [sosbasis[i][j]])
        end
        push!(constraints, Constraint(0,psd_dict,Dict(), samples))
    end
```
Since the positive semidefinite matrix variable ``Y`` occurs in every constraint, the corresponding cluster contains ``k \cdot length(samples)`` constraints after sampling. In order to split this into ``k`` clusters of ``length(samples)`` constraints, we use the option ``as_free`` to model ``Y`` as free variables:
```julia
    polprob = LowRankPolProblem(false,obj, constraints)
    sdp = ClusteredLowRankSDP(polprob; as_free = [:Y])
```
This adds auxilliary free variables ``X[i,j]``, adds the constraints ``X[i,j] = Y[i,j]``, and replaces the ``Y[i,j]`` in the constraints by ``X[i,j]``. Then the only positive semidefinite variables in the polynomial constraints are the sums-of-squares matrices, because of which each sums-of-squares constraint is assigned to its own cluster.

