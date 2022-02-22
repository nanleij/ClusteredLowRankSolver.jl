# Multivariate polynomial optimization using symmetry reduction
In this example, we consider minimizing a multivariate polynomial with ``S_3`` symmetries. The example is inspirated by example 7.1 from [gatermann-symmetry-2004](@cite). See the [Examples](https://github.com/nanleij/ClusteredLowRankSolver.jl/tree/main/examples) folder for the file with the code. We consider the polynomial
```math
f(x,y,z) = x^4 + y^4 + z^4 - 4xyz + x + y + z
```
for which we want to find the minimum value ``f_{min} = \min_{x,y,z} f(x,y,z)``. Relaxing the problem with a sum-of-squares constraint gives
```math
\begin{aligned}
    \max \quad& M &\\
    \text{s.t.} \quad& f - M &\text{is a sum-of-squares,}
\end{aligned}
```
where ``b`` is a vector of basis polynomials in variables ``x,y,z`` up to a certain degree ``d``.
Since the polynomial ``f`` is invariant under permuting the variables (i.e., the group action of ``S_3``), it is natural to consider only invariant sum-of-squares polynomials. From [gatermann-symmetry-2004](@cite), we know that any sum-of-squares polynomial invariant under the action of ``S_3`` can be written as
```math
    \langle Y_1, ww^{\sf T} \rangle + \langle Y_2, \Pi_2 \otimes ww^{\sf T} \rangle + \langle Y_3, \Pi_3 \otimes ww^{\sf T} \rangle
```
where ``w(x,y,z)`` is a vector of basis polynomials of the invariant ring ``\R[x,y,z]^{S_3} = \R[x+y+z, xy+yz+xz, xyz]``, and where ``\Pi_2(x,y,z) = ((x-y)(y-z)(z-x))^2`` and
```math
    \Pi_3 = \begin{pmatrix}2\phi_1^2-6\phi_2 & -\phi_1\phi_2+9\phi_3 \\ -\phi_1\phi_2+9\phi_3 & 2\phi_2^2- 6\phi_1\phi_3\end{pmatrix}.
```
In particular, ``\Pi_3`` is of rank 2 and has the decomposition ``v_1 v_1^{\sf T} + v_2 v_2^{\sf T} ``
with
```math
v_1 = \frac{1}{\sqrt{2}}\begin{pmatrix}
    2x-y-z \\
    2yz-zx-xy
\end{pmatrix} \quad \text{ and } \quad v_2 = \sqrt{\frac{3}{2}}\begin{pmatrix}
    y-z \\
    zx-xy
\end{pmatrix}
```
Since we consider sum-of-squares polynomials of a certain degree ``d``, we restrict to the elements of the matrices ``\Pi_i \otimes ww^{\sf T}`` with degree at most ``\lfloor d/2 \rfloor``.

To sample this constraint we need a three-variate minimal unisolvent set for invariant polynomials of degree at most ``d``. In this case, one such example are the representatives of the orbits of ``S_3`` of the rational points in the simplex with denominator ``d``, since these points are invariant under ``S_3`` and are minimal unisolvent (see [de-laat-clustered-2022](@cite)). However, if the polynomial space is more complicated it is unclear what a minimal unisolvent set is. To show one approach on this we instead make a grid of points which unisolvent but not minimal, and we use the approach of [de-laat-clustered-2022](@cite) through `approximatefekete` to choose a good subset of these points and a corresponding good basis for ``w``.

As for the example of the Delsarte bound, the objective is simply one free variable ``M`` with coefficient ``1``. We also create a function to generate an invariant basis with variables ``x,y,z``.
```julia
using ClusteredLowRankSolver, BasesAndSamples, AbstractAlgebra

function invariant_basis(x,y,z, d)
    # create a vector with a precise type
    v = [(x*y*z)^0]
    for deg=1:d, j=0:div(deg,3), i=0:div(deg-3j,2)
        # monomials in the invariant elementary polynomials
        # ordered by degree
        push!(v, (x+y+z)^(deg-2i-3j) * (x*y+y*z+z*x)^i * (x*y*z)^j)
    end
    return v
end

function min_f(d)

    obj = Objective(0, Dict(), Dict(:M => 1))
```
In this case, we need a three-variate polynomial ring, and a basis of invariant polynomials. We also use `approximatefekete` to find a subset of the sample points with a good basis.
```julia
    FF = RealField
    R, (x,y,z) = PolynomialRing(FF, ["x", "y", "z"])
    # The polynomial f:
    f =  x^4 + y^4 + z^4 - 4x*y*z + x + y + z

    # An invariant basis up to degree d:
    basis = invariant_basis(x, y, z, 2d)
    # For the sum-of-squares polynomials we have to
    # select elements of the basis based on the degree
    degrees = [total_degree(p) for p in basis]

    # generate samples and a good basis
    cheb_points = [vcat(sample_points_chebyshev(2d+k)...) for k=0:2]
    samples_grid = [[cheb_points[1][i+1], cheb_points[2][j+1], cheb_points[3][k+1]]
        for i=0:2d for j=0:2d+1 for k=0:2d+2]
    basis, samples = approximatefekete(basis, samples_grid)
```
Now we will construct the constraint matrices corresponding to the sum-of-squares parts. Although `approximatefekete` returns a polynomial basis only characterized by the evaluations on the sample points, we can work with it as if it were normal polynomials. Among others, this means that we can take the kronecker product of the sampled basis polynomials and a vector of polynomials.
```julia
    psd_dict = Dict()
    symmetry_weights = [[[R(1)]],
            [[R((x-y)*(y-z)*(z-x))]],
            [[1/sqrt(FF(2))*(2x-y-z),1/sqrt(FF(2))*(2y*z-x*z-x*y)],
                [sqrt(FF(3)/FF(2))*(y-z),sqrt(FF(3)/FF(2))*(x*z-x*y)]]]
    for swi=1:length(symmetry_weights)
        rank = length(symmetry_weights[swi])
        # create a decomposition of Π_i ⊗ ww^T in terms of polynomial vectors
        vecs = [kron(symmetry_weights[swi][r], basis) for r=1:rank]
        # This has in general too many entries,
        # so we will remove the ones with too high degree.
        for r=1:rank
            len = length(basis)
            # we keep the elements with degree at most d.
            keep_idx = [i for i=1:length(vecs[r])
                if total_degree(symmetry_weights[swi][r][div(i-1,len)+1]) +
                    degrees[(i-1)%len+1] <= d]
            vecs[r] = vecs[r][keep_idx]
        end
        psd_dict[Block((:trivariatesos,swi))] =
                LowRankMatPol([R(1) for r=1:rank], vecs)
    end
```
Now we can formulate the constraint and solve the `ClusteredLowRankSDP`:
```julia
    # the constraint is SOS + M = f
    constr = Constraint(f, psd_dict, Dict(:M => R(1)), samples)
    pol_problem = LowRankPolProblem(true, obj, [constr])
    sdp = ClusteredLowRankSDP(pol_problem)

    solvesdp(sdp)
end
```
