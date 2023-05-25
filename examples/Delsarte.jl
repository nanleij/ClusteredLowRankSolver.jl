module Delsarte

using AbstractAlgebra, ClusteredLowRankSolver, BasesAndSamples, QuadGK

export delsarte, Nspherical_cap_packing

function delsarte(n, d,costheta, precision=512; all_free = false, kwargs...)
    #set up the polynomial field
    setprecision(precision)
    FF = RealField
    P, (u, ) = PolynomialRing(FF, ["u"])

    #compute both the gegenbauer polynomials and the sos basis
    gbasis = basis_gegenbauer(2d, n, u)
    samples = sample_points_chebyshev(2d,-1,costheta)
    sosbasis, samples = approximatefekete([u^k for k=0:2d], samples)

    #construct the constraint ∑_k a_k P^n_k(u) + SOS + (u+1)(cos(θ)-u)*SOS = - 1
    c = Dict()
    for k=0:2d
        c[Block(k)] = LowRankMatPol([gbasis[k+1]], [[1]])
    end
    c[Block(:A)] = LowRankMatPol([1], [sosbasis[1:d+1]])
    c[Block(:B)] = LowRankMatPol([(u+1)*(costheta-u)], [sosbasis[1:d]])
    constraint = Constraint(-1, c, Dict(), samples)

    #Construct the objective 1 + ∑_k a_k
    objective = Objective(1, Dict(Block(k) => hcat([FF(1)]) for k=0:2d), Dict())

    #Construct the SOS problem: minimize the objective s.t. the constraint
    sos = LowRankPolProblem(false, objective, [constraint])

    #Construct the SDP with or without using free variables for the a_k
    if all_free
        sdp = ClusteredLowRankSDP(sos,as_free = [k for k=0:2d])
    else
        sdp = ClusteredLowRankSDP(sos)
    end

    #Solve the SDP and return results
    return status, sol,time, errorcode = solvesdp(sdp; kwargs...)
end
export delsarte_highrank
function delsarte_highrank(n, d,costheta, precision=512; all_free = false, kwargs...)
    #set up the polynomial field
    setprecision(precision)
    FF = RealField
    P, (u, ) = PolynomialRing(FF, ["u"])

    #compute both the gegenbauer polynomials and the sos basis
    gbasis = basis_gegenbauer(2d, n, u)
    samples = sample_points_chebyshev(2d,-1,costheta)
    sosbasis, samples = approximatefekete([u^k for k=0:2d], samples)

    #construct the constraint ∑_k a_k P^n_k(u) + SOS + (u+1)(cos(θ)-u)*SOS = - 1
    c = Dict()
    for k=0:2d
        c[Block(k)] = hcat([gbasis[k+1]])#LowRankMatPol([gbasis[k+1]], [[1]])
    end
    c[Block(:A)] = LowRankMatPol([1], [sosbasis[1:d+1]])
    c[Block(:B)] = LowRankMatPol([(u+1)*(costheta-u)], [sosbasis[1:d]])
    constraint = Constraint(-1, c, Dict(), samples)

    #Construct the objective 1 + ∑_k a_k
    objective = Objective(1, Dict(Block(k) => hcat([FF(1)]) for k=0:2d), Dict())

    #Construct the SOS problem: minimize the objective s.t. the constraint
    sos = LowRankPolProblem(false, objective, [constraint])

    #Construct the SDP with or without using free variables for the a_k
    if all_free
        sdp = ClusteredLowRankSDP(sos, as_free = [k for k=0:2d])
    else
        sdp = ClusteredLowRankSDP(sos)
    end

    #Solve the SDP and return results
    return status, sol,time, errorcode = solvesdp(sdp; kwargs...)
end

function weights(theta, n)
    #approximate the integral up to a certain precision. For odd n, this can be done exactly (for even n, there is a term √(1-y^2) in the integral, so thats more difficult)
    # exact answer uses hypergeometric 2F1 functions. For n odd>=3, these terminate. Not sure about convergence at x=1.
    # exact answer of integral: [x 2F1(1/2, (3 - n)/2, 3/2, x)]_cos(theta)^1
    # Wolframalpha:  at 1 it equals sqrt(pi) Gamma(n/2-1/2) Gamma(n/2) = 1/the factor in front
    if n==3
        return 1//2*(1-cos(theta)) #odd n give an easy integral (a polynomial in the integration variable)
    elseif n==4
        return (theta-cos(theta)*abs(sin(theta)))/pi #converted to the type of theta
    elseif n==5
        return 1//2-3//4*cos(theta)+1//4 * cos(theta)^3
    end
    #for the others we just compute the integral numerically
    integral =
        sqrt(BigFloat(pi))^(-1) * gamma(BigFloat(n) / 2) / gamma(BigFloat(n - 1) / 2) *
        quadgk(y -> (1 - y^2)^((n - 3) / 2), cos(theta), BigFloat(1), rtol = 10^(-40))[1] # -30 goes relatively fast, -50 is slow
    return integral
end#

function Nspherical_cap_packing(n,d,thetas,N = length(thetas),precision=precision(BigFloat);all_free = false,solver_kwargs...)
    #Construct the polynomial ring and the gegenbauer polynomials
    setprecision(precision)
    FF = RealField
    w = weights.(thetas,n)

    P, (u, ) = PolynomialRing(FF, ["u"])
    basis = basis_gegenbauer(2d, n, u)

    constraints = []
    for i=1:N
        for j=1:i
            # sum_k a_k,i,j P_k^n(x) <= - sqrt(w(theta_i)*w(theta_j)) for x in [-1, cos(theta_i+theta_j)]
            # With sum of squares polynomials this equals:
            # sum_k a_k,i,j P_k^n(x) + SOS + (x+1)*(cos(theta_i+theta_j)-x)* SOS = - sqrt(w(theta_i)*w(theta_j))
            c = Dict()
            for k=0:2d
                if i != j
                    # We set both sides of the matrix
                    c[Block(k,i,j)] = LowRankMatPol([1//2*basis[k+1]], [[1]])
                    c[Block(k,j,i)] = LowRankMatPol([1//2*basis[k+1]], [[1]])
                else
                    c[Block(k,i,i)] = LowRankMatPol([basis[k+1]], [[1]])
                end
            end
            c[Block((:SOS1,i,j))] = LowRankMatPol([1], [[u^k for k=0:d]])
            c[Block((:SOS2,i,j))] = LowRankMatPol([(u+1)*(cos(thetas[i]+thetas[j])-u)], [[u^k for k=0:d-1]])

            samples = sample_points_chebyshev(2d,-1,cos(thetas[i]+thetas[j])) # TODO possibly take more samples
            push!(constraints, Constraint(-sqrt(w[i]*w[j]), c, Dict(), samples))
        end

        # The objective is M, but constrained by ∑_k a_kii ≦  M - w_i, i.e. ∑_k a_kii + pos.slack_i - M = -w_i
        obj = Dict(Block(k,i,i) => LowRankMatPol([1],[[1]]) for k=0:2d)
        #an easy mistake would be to use the same slack variable for every i by forgetting the i in the tuple (:SOS_obj, i).
        obj[Block((:SOS_obj,i))] = LowRankMatPol([1],[[1]])
        objective_constraint = Constraint(-w[i],obj,Dict(:M=>-1))
        push!(constraints,objective_constraint)
    end

    #The objective is M
    objective = Objective(FF(0), Dict(), Dict(:M => FF(1)))

    #Construct the SOS - problem, minimizing the objective
    sos = LowRankPolProblem(false, objective, constraints)

    #construct and solve the SDP, with or without using free variables for the A_k
    if all_free
        sdp = ClusteredLowRankSDP(sos,as_free=[k for k=0:2d])
    else
        sdp = ClusteredLowRankSDP(sos)
    end
    status, sol, time, errorcode = solvesdp(sdp;solver_kwargs...)
end


end # end module
