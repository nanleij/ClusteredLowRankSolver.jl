module Delsarte

using AbstractAlgebra, ClusteredLowRankSolver

export delsarte

function delsarte(n, d, costheta)
    # Initialize the objective with additive constant 0,
    # no dependence on matrix variables,
    # and dependence with coefficient 1 on the free variable :M
    obj = Objective(0, Dict(), Dict(:M => 1))

    R, x = polynomial_ring(RealField, :x)

    # 2d+1 Chebyshev points in the interval [-1, cos(θ)]
    samples = sample_points_chebyshev(2d, -1, costheta)
    # A basis for the Chebyshev polynomials of degree at most 2d
    basis = basis_chebyshev(2d, x)
    # Optimize the basis with respect to the samples
    sosbasis, samples = approximatefekete(basis, samples)


    # A vector consisting of the Gegenbauer polynomials
    # of degree at most 2d
    gp = basis_gegenbauer(2d, n, x)
    psd_dict1 = Dict()
    for k = 1:2d
        # the syntax [x;;] to create a matrix is not compatible with julia 1.6
        psd_dict1[(:a, k)] = hcat([gp[k+1]])
    end

    psd_dict1[(:SOS, 1)] = LowRankMatPol([1], [sosbasis[1:d+1]])
    psd_dict1[(:SOS, 2)] = LowRankMatPol([(1+x)*(costheta-x)], [sosbasis[1:d]])
    constr1 = Constraint(-1, psd_dict1, Dict(), samples)

    # The free variable M has coefficient -1:
    free_dict2 = Dict(:M => -1)
    # The variables a_k and the slack variable have coefficient 1:
    psd_dict2 = Dict()
    for k = 1:2d
        psd_dict2[(:a, k)] = hcat([1]) 
    end
    psd_dict2[:slack] = hcat([1])
    constr2 = Constraint(-1, psd_dict2, free_dict2)

    problem = Problem(Minimize(obj), [constr1, constr2])
    status, dualsol, primalsol, time, errorcode = solvesdp(problem)
    return objvalue(problem, primalsol)
end

end # end module