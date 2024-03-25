module DelsarteExact

using Nemo, ClusteredLowRankSolver

export delsarte_exact, delsarte_round

function delsarte_exact(n, d, costheta; FF=QQ, g=1, eps=1e-40, kwargs...)
    constraints = []

    P, x = polynomial_ring(FF, :x)

    gbasis = basis_gegenbauer(2d, n, x)
    sosbasis = basis_chebyshev(2d, x)

    samples = sample_points_chebyshev(2d)
    # round the samples to QQ:
    samples = [round(BigInt, x * 10^4)//10^4 for x in samples]

    c = Dict()
    for k = 0:2d
        c[k] = [gbasis[k+1];;]
    end
    c[:A] = LowRankMatPol([1], [sosbasis[1:d+1]])
    c[:B] = LowRankMatPol([(x+1)*(costheta-x)], [sosbasis[1:d]])
    push!(constraints, Constraint(-1, c, Dict(), samples))

    objective = Objective(1, Dict(k => [1;;] for k=0:2d), Dict())

    problem = Problem(Minimize(objective), constraints)

    problem_bigfloat = map(x->generic_embedding(x, g), problem)
    status, primalsol, dualsol, time, errorcode = solvesdp(problem_bigfloat; duality_gap_threshold=eps, kwargs...)

    return objvalue(problem, dualsol), problem, primalsol, dualsol
end

function delsarte_round(n, d, costheta; eps=1e-40, prec=512)
    obj, problem, primalsol, dualsol = delsarte_exact(n, d, costheta; eps=eps, prec=prec)
    # use monomial basis
    R, x = polynomial_ring(QQ, :x)
    b = [x^k for k=0:2d]
    success, exactdualsol = exact_solution(problem, primalsol, dualsol, monomial_bases=[b])
    return success, problem, exactdualsol
end

end # end module