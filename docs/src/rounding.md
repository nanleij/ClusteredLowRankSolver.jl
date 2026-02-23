# [Rounding numerical solutions to exact optimal solutions](@id secrounding)

In certain situations there are reasons to believe that there is a 'nice' optimal solution. That is, the kernel of every optimal solution can be defined over a number field of low algebraic degree and low bit size. In that case, we can use the rounding procedure from [CLL24](@cite) to obtain such a solution from a precise enough solution returned by a semidefinite programming solver. This is implemented in the function [`exact_solution`](@ref). 

See also the following examples:
- [A very basic example of minimizing a univariate polynomial](@ref rounding_univariate)
- [A more involved example of minimizing a multivariate polynomial](@ref Tutorial)
- [Rounding the Delsarte LP bound](@ref exrounding) 

```@docs
exact_solution
```

## A short introduction to the rounding procedure

The rounding procedure consists of three steps. First a nice basis for the kernel of the provided solution is found; this defines the optimal face. In the second step, this basis is used as part of a basis transformation, to obtain a smaller semidefinite program where the affine hull of the constraints has a one-to-one correspondence with the original optimal face. The provided solution is also transformed, to obtain a strictly feasible solution of the new semidefinite program. In the last step, we take an approximation of the numerical, transformed solution over the rationals (or a provided number field), and find an exact solution close to it. See [CLL24](@cite) for a more detailed description. 

## Settings for the rounding procedure

For ease of use, all settings which can be used to tweak the rounding procedure are collected in the [`RoundingSettings`](@ref) structure.
```@docs
RoundingSettings
```

### [Finding the right settings](@id roundingsettings)
Depending on the problem, the default parameters will suffice. In the following cases small changes are needed:
  1. The kernel vectors are not found correctly, or have significantly higher maximum number after reduction than the maximum numerator and denominator.
     - Try to solve to a smaller duality gap, and/or decrease the setting `kernel_round_errbound`.
     - Try the settings `kernel_use_dual=false`.
     
  2. The final solution does not satisfy the constraints. (Or not enough pivots were found) 
     - increase `redundancyfactor`, or set it to `-1` to take all variables into account.
     - This might also be an indication that the kernel vectors are not found correctly (see item 1).
     
  3. The final solution is not positive semidefinite.
     - Increase the setting `approximation_decimals`. Make sure that the provided solution has at least that many digits correct (the duality gap should be lower than `10^(-approximation_digits)`)
     - Increase the setting `pseudo_columnfactor`. The higher this setting, the closer the exact solution will be to the provided solution. However, this also increases the bit size of the exact solution.
     - In some cases this is also an indication that the kernel vectors are not found correctly (see item 1), especially when the reported negative eigenvalues are close to zero.

   4. I want a solution with smaller numbers.
     - Use `unimodular_transform=false`. This will give a non-unimodular transform, and it depends a lot on the kernel vectors whether this is better or worse.
     - Decrease the setting `approximation_decimals`. 
     - Use the setting `pseudo=false`. When this does give a solution, it typically has smaller bitsize.
     - If `pseudo=false` does not work, decrease `pseudo_columnfactor` instead. This gives something closer to `pseudo=false` while keeping the solution positive semidefinite.

## Using coefficient matching

Although the semidefinite program used in the solver is defined using sampling, it is possible to use a semidefinite program defined using coefficient matching in a monomial basis for the rounding procedure. This is not yet fully automated; the user needs to provide the monomial basis for each polynomial constraint to the rounding procedure in order to use this. Using coefficient matching instead of sampling in the rounding heuristic generally decreases the size of the exact solutions. See below for an example using the slightly modified code for the Delsarte LP bound in the [example](@ref exrounding) for rounding. Here we have one univariate constraint with polynomials up to degree ``2d``.
```julia
d = 3
problem, dualsol, primalsol = delsarte(8, 1//2, d; duality_gap_threshold=1e-30)
R, (x,) = polynomial_ring(QQ, 1)
mon_basis = [x^k for k=0:2d]
success, exactdualsol = exact_solution(problem, dualsol, primalsol, monbases = [mon_basis])
```

## Finding the appropriate number field for the rounding procedure

It is in general not directly clear over which number field the optimal face can be defined. The function [`find_field`](@ref) can help to find such a field. See Section 2.5 of [CLL24](@cite) for an explanation of the procedure. Note that this is a heuristic, and might not find the correct field (especially if the solution is of relatively low precision compared to the degree of the number field).
```@docs
find_field
```

## Getting an exact feasible solution close to the optimum

In general, the kernel of the optimal solutions can only be defined using a number field of high algebraic degree, or with large bit size. In this case it is unlikely that the exact kernel can be found, so that the rounding procedure will fail. However, it is possible to get an exact feasible solution with objective value close to the optimum using the rounding procedure. To do this, solve the semidefinite program as a feasibility problem (no objective), with the extra constraint that the (original) objective should be a rational number close to the optimum value. Then the solver will return a strictly feasible solution if it exists (that is, with empty kernel). The function [`exact_solution`](@ref) will now essentially skip the steps of finding the kernel vectors and transforming the problem, since there are no kernel vectors. Therefore, it will find an exact feasible solution close to the provided solution such that the objective of the original problem is equal to the rational number in the extra constraint. See [Rounding the Delsarte LP bound](@ref exrounding) for a worked example.


