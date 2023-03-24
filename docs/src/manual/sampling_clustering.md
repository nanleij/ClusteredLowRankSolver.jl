# Sampling and Clustering

## Sampling
For the samples, we need a minimal unisolvent set of points. A set of points is unisolvent for a space of polynomials if the only polynomial which evaluates to ``0`` on all the points is the zero polynomial. Such a set is minimal if any strict subset is not unisolvent. In the univariate case, any `d+1` distinct points are unisolvent for polynomials up to degree `d`. In general, it is possible to test whether a set of points is unisolvent by creating the Vandermonde matrix ``(b_j(x_i))_{ij}`` and checking whether it is nonsingular. This matrix is also used by [`approximatefekete`](@ref) to select a subset of good sample points and a correspondingly good basis. Thus a method to create a unisolvent set of points is to create a grid of points, or take random points, and use [`approximatefekete`](@ref).

## Clustering
In the [`ClusteredLowRankSDP`](@ref), the constraints are clustered. That is, two constraints in different clusters do not use the same positive semidefinite matrix variables. Internally, this creates a block structure which can be exploited by the solver. One example where clustering occurs is when sampling multiple polynomial constraints which do not use positive semidefinite variables other than the ones for sum-of-squares characterizations. If other positive semidefinite variables are used, it might be beneficial to use extra free variables in the constraint instead of the positive semidefinite matrix variables, and add constraint to equate these variables to entries of the positive semidefinite matrix variables. This is supported with the keyword argument `as_free`:
 ```julia
 	sdp = ClusteredLowRankSDP(pol_prob, as_free = [:A, :B,  ...])
```
where `:A, :B, ...` are the keys in the `Block` structure corresponding to the positive semidefinite matrix variables which should be modelled using extra free variables.
See [Clustering](@ref) for an example. This option can also be used to avoid high-rank constraint matrices. 
