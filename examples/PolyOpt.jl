module PolyOpt

using ClusteredLowRankSolver, AbstractAlgebra

export polyopt, min_f

function polyopt(f, d, u)
    #set up the polynomial field

    #compute the sos basis and the samples
    sosbasis = basis_chebyshev(d, u)
    samples = sample_points_chebyshev(2d,-1,1)

    #construct the constraint f - lambda = SOS 
    c = Dict()
    c[(:sos, 1)] = LowRankMatPol([1], [sosbasis[1:d+1]])
    constraint = Constraint(f, c, Dict(:lambda => 1), samples)

    #Construct the objective 1 + ∑_k a_k
    objective = Objective(0, Dict(), Dict(:lambda => 1))

    #Construct the SOS problem: minimize the objective s.t. the constraint
    problem = Problem(true, objective, [constraint])

    sdp = ClusteredLowRankSDP(problem)
    sdp = convert_to_prec(sdp)
    #Solve the SDP and return results
    status, dualsol, primalsol, time, errorcode = solvesdp(sdp)    
    return objvalue(problem, primalsol), freevar(primalsol, :lambda), primalsol
end


function invariant_basis(x,y,z, d)
    # create a vector with a precise type
    v = [(x+y+z)^(deg-2i-3j) * (x*y+y*z+z*x)^i * (x*y*z)^j for deg=0:d for j=0:div(deg,3) for i=0:div(deg-3j,2)]
    return v
end

# example of S_3 invariance
function min_f(d)
    obj = Objective(0, Dict(), Dict(:M => 1))

    FF = RealField
    R, (x,y,z) = polynomial_ring(FF, [:x, :y, :z])
    # The polynomial f:
    f =  x^4 + y^4 + z^4 - 4x*y*z + x + y + z

    # An invariant basis up to degree d:
    basis = invariant_basis(x, y, z, 2d)
    # For the sum-of-squares polynomials we have to
    # select elements of the basis based on the degree
    degrees = [total_degree(p) for p in basis]

    # generate samples and a good basis
    cheb_points = [sample_points_chebyshev(2d+k) for k=0:2]
    samples_grid = [[cheb_points[1][i+1], cheb_points[2][j+1], cheb_points[3][k+1]]
        for i=0:2d for j=0:2d+1 for k=0:2d+2]
    basis, samples = approximatefekete(basis, samples_grid)

    psd_dict = Dict()
    equivariants = [[[R(1)]], [[(x-y)*(y-z)*(z-x)]], [[(2x-y-z), (2y*z-x*z-x*y)], [(y-z), (x*z-x*y)]]]

    factors = [[1], [1], [1//2, 3//2]]
    for eqi in eachindex(equivariants)
        vecs = []
        for r in eachindex(equivariants[eqi])
            vec = []
            for eq in equivariants[eqi][r], (q, qdeg) in zip(basis, degrees)
                if 2total_degree(eq) + 2qdeg <= 2d
                    push!(vec, eq * q)
                end
            end
            if length(vec) > 0
                push!(vecs, vec)
            end
        end
        if length(vecs) > 0
            psd_dict[(:trivariatesos, eqi)] = LowRankMatPol(factors[eqi], vecs)
        end
    end

    constr = Constraint(f, psd_dict, Dict(:M => 1), samples)
    problem = Problem(Maximize(obj), [constr])

    status, dualsol, primalsol, time, errorcode = solvesdp(problem)
    return problem, dualsol, primalsol
end

end # end module