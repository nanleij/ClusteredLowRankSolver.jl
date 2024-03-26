module ThreePointBound

using Random, ClusteredLowRankSolver, Nemo

export three_point_spherical_codes

function Q(FF, n::Int, k::Int, u, v, t)
    R, x = polynomial_ring(FF, :x)
    p = basis_gegenbauer(k, n, x)[end]
    sum(coeff(p, i) * ((1-u^2)*(1-v^2))^div(k-i, 2)*(t-u*v)^i for i=0:length(p)-1)
end

function Smat(FF, n, k, d, u, v, t)
    mat =  Q(FF, n-1, k, u, v, t) .* (m(v, d-k) * transpose(m(u, d-k)) + m(u, d-k) * transpose(m(v, d-k)))
    mat += Q(FF, n-1, k, t, u, v) .* (m(t, d-k) * transpose(m(u, d-k)) + m(u, d-k) * transpose(m(t, d-k)))
    mat += Q(FF, n-1, k, t, v, u) .* (m(t, d-k) * transpose(m(v, d-k)) + m(v, d-k) * transpose(m(t, d-k)))
    1//6 * mat
end

function m(w, d)
    [w^k for k=0:d]
end

p(u, costheta) = p(u, -1, costheta)
p(u, a, b) = (u-a)*(b-u)

"""
    three_point_spherical_codes(n, costheta, d2, d3, kwargs])

Compute an upper bound on the number of caps of size costheta that fit on the sphere S^{n-1}.
The generated SDP uses the uvt S_3 symmetry.

# Arguments
- `n`: the dimension
- `costheta`: the size of the spherical caps.
- `d2`: half the degree of the polynomials used to model the one variable part of the function.
        Use -1 to disable this completely.
- `d3`: half the degree of the polynomials used to model the three variable part of the function.
        Use -1 to disable this completely.
- `N2=max(d2,d3)`: half the degree used for the univariate SOS certificate.
- `N3=d3`: half the degree used for the trivariate SOS certificate.
- `prec`: the precision in bits used to setup and solve the SDP.
- `kwars': additional arguments to give to the solver.
"""
function three_point_spherical_codes(
        n,
        costheta,
        d2,
        d3;
        FF=QQ,
        kwargs...)
    N2 = max(d2, d3)
    N3 = d3

    Random.seed!(1935) # make the samples reproducible

    constraints = []

    @info "Creating the univariate constraint"
    W, w = polynomial_ring(FF, :w)

    f = Dict()
    for k=0:d3
        Tmat = 3Smat(FF, n, k, d3, w, w, 1)
        f[(:F, k)] = Tmat
    end
    if d2 >= 0
        basis = basis_gegenbauer(2d2, n, w)
        for k = 0:2d2
            f[(:a, k)] = LowRankMatPol([basis[k+1]], [[1]])
        end
    end
    basis1d = basis_chebyshev(2N2, w)
    samples1d = sample_points_chebyshev(2N2, -1, 1)
    samples1d = [floor(BigInt, 10^4*x)//10^4 for x in samples1d]

    if N2 >= 0
        f[(:univariatesos, 1)] = LowRankMatPol([1], [basis1d[1:N2+1]])
    end
    if N2 >= 1
        f[(:univariatesos, 2)] = LowRankMatPol([p(w, costheta)], [basis1d[1:N2]])
    end

    push!(constraints, Constraint(-1, f, Dict(), samples1d))

    @info "Constructing trivariate constraint"
    # F(u,v,t)  + weighted sos = 0

    R, (u, v, t) = polynomial_ring(FF, [:u, :v, :t])

    equivariants = [[[R(1)]], [[(u-v)*(v-t)*(t-u)]], [[2u-v-t, 2v*t-u*t-u*v], [v-t, u*t-u*v]]]

    factors = [[1], [1], [1//2, 3//2]]

    weights = [
        R(1),
        p(u,costheta)+p(v,costheta)+p(t,costheta),
        p(u,costheta)*p(v,costheta)+p(v,costheta)*p(t,costheta)+p(t,costheta)*p(u,costheta),
        p(u,costheta)*p(v,costheta)*p(t,costheta),
        2u*v*t + 1 - u^2 - v^2 - t^2]

    tmp = length([deg for deg=0:2N3 for k=0:div(deg, 3) for j=0:div(deg-3k, 2)])
    cheb_points = [vcat(sample_points_chebyshev(2N3+k, -1, 1)...) for k=0:2]
    samples = [[cheb_points[1][i+1], cheb_points[2][j+1], cheb_points[3][k+1]] for i=0:2N3 for j=0:2N3+1 for k=0:2N3+2]
    samples = sort(samples[shuffle(1:length(samples))[1:tmp]])
    samples = [[QQ(floor(BigInt, 10^4*x))//10^4 for x in sample] for sample in samples]

    # from now on we work with sampled polynomials
    R = SampledMPolyRing(FF, samples)
    u = R(u)
    v = R(v)
    t = R(t)

    F = Dict()
    for k=0:d3
        Tmat = Smat(FF, n, k, d3, u, v, t)
        F[(:F, k)] = Tmat
    end

    _, x = polynomial_ring(FF, :x)
    tempbasis = m(x, N3)

    basis3d = []
    for deg = 0:N3, k = 0:div(deg, 3), j = 0:div(deg-3k, 2)
        evaluations = [tempbasis[deg-3k-2j+1](u+v+t) * tempbasis[j+1](u*v+v*t+u*t) * tempbasis[k+1](u*v*t) for (u, v, t) in samples]
        push!(basis3d, SampledMPolyRingElem(R, evaluations))
        # other possibility:
        # push!(basis3d, tempbasis[deg-3k-2j+1](u+v+t) * tempbasis[j+1](u*v+v*t+u*t) * tempbasis[k+1](u*v*t))
    end
    # degrees of the sampled polynomials
    degrees3d = [deg for deg=0:N3 for k=0:div(deg, 3) for j=0:div(deg-3k, 2)]

    # create the invariant sums-of-squares
    for wi in eachindex(weights)
        if total_degree(weights[wi]) <= 2N3
            for eqi in eachindex(equivariants)
                vecs = []
                for r in eachindex(equivariants[eqi])
                    vec = []
                    for eq in equivariants[eqi][r], (q, qdeg) in zip(basis3d, degrees3d)
                        if total_degree(weights[wi]) + 2total_degree(eq) + 2qdeg <= 2N3
                            push!(vec, eq * q)
                        end
                    end
                    if length(vec) > 0
                        push!(vecs, vec)
                    end
                end
                if length(vecs) > 0
                    F[(:trivariatesos, wi, eqi)] = LowRankMatPol(weights[wi] .* factors[eqi], vecs)
                end
            end
        end
    end
    push!(constraints, Constraint(0, F, Dict(), samples))
    
    # create the objective
    objdict = Dict()
    objdict[(:F, 0)] = ones(Int, d3+1, d3+1)
    for k=0:2d2
        objdict[(:a, k)] = ones(Int, 1, 1)
    end
    obj = Objective(1, objdict, Dict())
    problem = Problem(Minimize(obj), constraints)

    # solve the problem
    status, primalsol, dualsol, time, errorcode = solvesdp(problem; kwargs...)
    return problem, primalsol, dualsol
end

end # of module