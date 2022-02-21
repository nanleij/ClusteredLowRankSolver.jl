# Example: Binary sphere packing
This is a slightly more advanced example, which uses the non-diagonal block structure of the constraint matrices. We consider the maximum density which can be obtained by packing spheres with radii ``r_1, \ldots, r_N`` in ``\R^n``. In [de-laat-upper-2014](@cite) the Cohn-Elkies linear programming bound for sphere packing densities [cohn-new-2003](@cite) is generalized to ``N`` radii. We will follow their approach to define an optimization problem in terms of polynomial equality constraints.

!!! compat "Theorem"
    Let ``r_1,\ldots,r_N>0``, and let ``f:\mathbb{R}^n \to \mathbb{R}^{N \times N}`` be a matrix-valued function whose every component ``f_{ij}`` is a radial Schwartz function. Suppose ``f`` satisfies the following conditions:
      1. The matrix ``\big(\hat{f}_{ij}(0)- (\mathrm{vol}\, B(r_i))^{1/2}(\mathrm{vol}\, B(r_j))^{1/2}\big)_{i,j=1}^N`` is positive semidefinite, where ``B(r)`` is the ball of radius ``r`` centered at the origin.
      2. The matrix of Fourier transforms ``\big(\hat{f}_{ij}(t)\big)_{i,j=1}^N`` is positive semidefinite for every ``t>0``.
      3. ``f_{ij}(w)\leq 0`` if ``w \geq r_i+r_j``, for ``i,j = 1,\ldots,N``.
    Then the density of any sphere packing of spheres of radii ``r_1,\ldots,r_N`` in the Euclidean space ``\mathbb{R}^n`` is at most ``\max \{f_{ii}(0): i=1,\ldots,N\}.``

Using functions of the form
```math
\widehat{f}(x) = \sum_{k=0}^d A_k \|x\|^{2k} e^{-\pi \|x\|^2},
```
we obtain by Lemma 5.3 of [de-laat-upper-2014](@cite)
```math
f(x) = \sum_{k=0}^d A_k \frac{k!}{\pi^k}L_k^{n/2-1}(\pi \|x\|^2)e^{-\pi \|x\|^2},
```
where ``A_k`` are real, symmetric ``N \times N`` matrices, and ``L_k^{n/2-1}`` is the degree ``k`` Laguerre polynomial with parameter ``n/2-1``. Since positive factors do not influence positivity, we can ignore the factors ``e^{-\pi \|x\|^2}``. We denote the resulting polynomials still by ``f`` and ``\hat{f}``, abusing notation. This gives the constraint for first requirement of the Theorem
```math
\hat{f}(0) - \big((\mathrm{vol}\, B(r_i))^{1/2}(\mathrm{vol}\, B(r_j))^{1/2}\big)_{i,j=1}^N = Y_0 \succeq 0.
```
We can apply this constraint element-wise to get ``N(N-1)/2`` constraints enforcing the first requirement.

Using the theory of matrix polynomial programs (see e.g. [de-laat-clustered-2022](@cite)), we know that the second requirement is fulfilled if and only if there are positive semidefinite matrices ``Y_1`` and ``Y_2`` such that
```math
\hat{f}(t)_{ij} = \langle b(t)b(t)^{\sf T} \otimes E_{ij}, Y_1 \rangle + \langle tb(t)b(t)^{\sf T} \otimes E_{ij}, Y_2 \rangle
```
for every entry ``\hat{f}_{ij}``, ``i,j=1,\ldots, N``.

We can enforce the third requirement with the constraints that there are positive semidefinite matrices ``Z_{ij}^l`` for ``l=1,2`` and ``i,j=1,\ldots,N`` such that
```math
f(w)_{ij} + \langle b(w)b(w)^{\sf T}, Z_{ij}^1 \rangle + \langle (w-(r_i+r_j)^2)b(w)b(w)^{\sf T}, Z_{ij}^2 \rangle = 0
```
where we denote ``w = \|x\|^2`` (note that ``f(x)_{ij}`` is a polynomial in ``\|x\|^2``).

Finally, the objective can be modelled by introducing a free variable ``M`` and requiring that
```math
f_{ii}(0) \leq M
```
where we minimize over ``M``. This assures that ``M = \max\{f_{ii}(0)\}``, which equals the bound given by the function ``f``.

In the following, we will show how to model the second type of constraint. This constraint contains the internal block structure, which is not present in the previous example of the Delsarte linear programming bound. For this, we assume we defined a certain maximum degree `d` and the dimension `n`.
We use the packages `AbstractAlgebra`, `ClusteredLowRankSolver` and `BasesAndSamples`. As in the previous example, we first construct a suitable basis and sample points.
```julia
R, (x,) = PolynomialRing(RealField, ["x"])

basis = basis_laguerre(2d+1, BigFloat(n)/2-1, BigFloat(2*pi)*x)
samples = sample_points_rescaled_laguerre(2d+1)
basis, samples = approximatefekete(basis, samples)
```
Now, for given `i != j` (over which we normally would loop to get all constraints), we can start defining the corresponding constraint.
```julia
free_dict = Dict()
for k = 0:2d+1
    # Although we don't need to use the Block structure,
    # we can of course do it if it is more convenient.
    # Instead, we could e.g. use (:A, k, i, j) as key
    free_dict[Block(k, i, j)] = -x^k
end

psd_dict = Dict()
# We need to keep the blocks symmetric,
# so we distribute the matrices over the two blocks.
psd_dict[Block(:SOS21, i, j)] = LowRankMatPol([R(1//2)], [basis[1:d+1]])
psd_dict[Block(:SOS21, j, i)] = LowRankMatPol([R(1//2)], [basis[1:d+1]])

psd_dict[Block(:SOS22, i, j)] = LowRankMatPol([1//2*x], [basis[1:d+1]])
psd_dict[Block(:SOS22, j, i)] = LowRankMatPol([1//2*x], [basis[1:d+1]])

constr = Constraint(R(0), psd_dict, free_dict, samples)
```
For `i = j`, the definition is similar, except that we do not need to distribute the constraint matrices over two blocks. See the [Examples](https://github.com/nanleij/ClusteredLowRankSolver.jl/tree/main/examples) folder for the full code.
