# Advanced modeling

Here some more advanced modeling concepts are explained. Please see the [Tutorial](@ref) for basic usage.

## Block structure
In some cases, there is low-rank structure in the constraint matrices, but it is only obvious when considering submatrices. This is for example the case when using sums-of-squarse for polynomial matrix constraints (see the [sphere packing example](@ref exspherepacking)). In these situations the [`Block`](@ref) struct can be used to specify to which subblock the given constraint matrix corresponds. In all versions of `ClusteredLowRankSolver`, only a subblock structure with equal sized blocks is allowed.
```@docs
Block
```
!!! note 
    For version 0.x, using the `Block` struct was required for specifying the names of the positive semidefinite variables. This is no longer the case in version 1.0+.


## Sampling
For the samples, we need a minimal unisolvent set of points for the polynomial vector space spanned by the entries of the coefficients in the constraint. A set of points is unisolvent for a space of polynomials if the only polynomial which evaluates to ``0`` on all the points is the zero polynomial. Such a set is minimal if any strict subset is not unisolvent. In general, it is possible to test whether a set of points is unisolvent by creating the Vandermonde matrix ``(b_j(x_i))_{ij}`` and checking whether it is nonsingular. This matrix is also used by [`approximatefekete`](@ref) to select a subset of good sample points and a correspondingly good basis. 

There are three main approaches to create a unisolvent set of points. First, one can take a number of random samples (at least equal to the dimension of the polynomial space considered in the constraint). This is a unisolvent set with probability 1, and by using the function [`approximatefekete`](@ref) it is possible to select a minimal unisolvent subset of these points, together with a good basis. The second approach is to take a grid of points, and again use [`approximatefekete`](@ref) to select a minimal unisolvent subset.  Lastly, one can use a set known to be unisolvent, such as the chebyshev points for univariate polynomials ([`sample_points_chebyshev`](@ref)), padua points for bivariate polynomials ([`sample_points_padua`](@ref)), or the rational points in the simplex with denominator ``1/d`` for multivariate polynomials of degree at most `d` ([`sample_points_simplex`](@ref)).

```@docs; canonical=false
approximatefekete
approximatefeketeexact
sample_points_chebyshev
sample_points_padua
sample_points_simplex
```

## Sampled polynomials
Sampling large polynomials can become the bottleneck of building the semidefinite program. To speed this up, it is possible to work with sampled polynomials. Functionality includes basic operations such as multiplication, addition and subtraction, but also evaluating a polynomial on a sampled polynomial. Below we show some examples of how such a sampled polynomial can be constructed.
```@example
using AbstractAlgebra, ClusteredLowRankSolver
R, x = polynomial_ring(RealField, :x)
samples = sample_points_chebyshev(10)
# The samples need to be sorted for the sampled polynomial ring for faster evaluations
sort!(samples) 
Rsampled = sampled_polynomial_ring(RealField, samples)
# options to construct a sampled polynomial:
# (partially) construct a polynomial and convert it to a sampled polynomial
p = 1+x^2 
psampled = Rsampled(p)
# convert the generator and work with that
xs = Rsampled(x)
psampled2 = 1+xs^2
# evaluate a polynomial on a sampled polynomial
psampled3 = p(xs)
# construct the polynomial from evaluations:
evals = [1+i^2 for i in samples]
psampled4 = SampledMPolyRingElem(Rsampled, evals)
length(unique([psampled, psampled2, psampled3, psampled4])) == 1
```
Note that it should be possible to evaluate polynomials over the base ring of the sampled polynomial ring on the samples. For example, it is not possible to use floating point samples with an exact base ring.
A sampled polynomial can only be evaluated at the samples used in the sampled polynomial ring.
```@docs
sampled_polynomial_ring
SampledMPolyRingElem
``` 


## Clustering
In the [`ClusteredLowRankSDP`](@ref), the constraints are clustered. That is, two constraints in different clusters do not use the same positive semidefinite matrix variables. Internally, this creates a block structure which can be exploited by the solver. One example where clustering occurs is when sampling multiple polynomial constraints which do not use positive semidefinite variables other than the ones for sum-of-squares characterizations. If other positive semidefinite variables are used, it might be beneficial to use extra free variables in the constraint instead of the positive semidefinite matrix variables, and add constraint to equate these variables to entries of the positive semidefinite matrix variables. This is supported through the function [`model_psd_variables_as_free_variables!`](@ref). See the [example](@ref exClustering) about Clustering. 
```@docs
model_psd_variables_as_free_variables!
```




