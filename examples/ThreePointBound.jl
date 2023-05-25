module ThreePointBound

using ClusteredLowRankSolver, AbstractAlgebra, BasesAndSamples

export three_point_spherical_cap


function Q(n::Int, k::Int, u, v, t)
    R, x = PolynomialRing(RealField, "x")
    p = basis_gegenbauer(k, n, x)[end]
    sum(coeff(p, i) * ((1-u^2)*(1-v^2))^div(k-i, 2)*(t-u*v)^i for i=0:length(p)-1)
end

function m(w, d)
    [w^k for k=0:d]
end

p(u, costheta) = (u+1)*(costheta-u)

function three_point_spherical_cap(n, d, costheta, prec=256, all_free=false; basis=1, grid_option=2, N=2d, verbose=false, kwargs...)
    #basis: 1 = monomial
    #       2 = chebyshev
    #       3 = gegenbauer
    # These are orthogonalized with respect to the sample points, so it shouldn't really matter
    setprecision(BigFloat,prec)
    FF = RealField

    R, (u,v,t) = PolynomialRing(FF, ["u","v","t"])
    if verbose
        println("Creating the polynomials for F_k...")
    end
    F = Dict()
    for k=0:d
        #Instead of having both e.g. m(v,d-k) * m(u,d-k)^T and m(t,d-k) * m(u,d-k)^T, we combine them
        F[Block((:F,k))] = LowRankMatPol([R(1) for i=1:3],
                                    [m(u,d-k), m(v,d-k), m(t,d-k)], #combine the vectors which are used multiple times
                                    [1//6*(Q(n-1, k, u, v, t) .* m(v,d-k) + Q(n-1, k, u, t, v) .* m(t,d-k)),
                                    1//6*(Q(n-1, k, v, u, t) .* m(u,d-k) + Q(n-1, k, v, t, u) .* m(t,d-k)),
                                    1//6*(Q(n-1, k, t, v, u) .* m(v,d-k) + Q(n-1, k, t, u, v) .* m(u,d-k))])
    end
    #The vectors which construct the Pi_pi matrices. We have to tensor them with a symmetric basis for the E_pi matrices
    symmetry_weights = [[[R(1)]],
                        [[(u-v)*(v-t)*(t-u)]],
                        [[1/sqrt(FF(2))*(2u-v-t),1/sqrt(FF(2))*(2v*t-u*t-u*v)],
                            [sqrt(FF(3)/FF(2))*(v-t),sqrt(FF(3)/FF(2))*(u*t-u*v)]]]
    #The polynomial weights which describe the domain
    weights = [R(1),
        p(u,costheta)+p(v,costheta)+p(t,costheta),
        p(u,costheta)*p(v,costheta)+p(v,costheta)*p(t,costheta)+p(t,costheta)*p(u,costheta),
        p(u,costheta)*p(v,costheta)*p(t,costheta),
        2u*v*t + 1 - u^2 - v^2 - t^2]
    #The basis in 1 variable used for the symmetric basis
    R1, (x,) = PolynomialRing(FF,["x"])
    if basis == 1
        basis1d_sym = m(x, 2d)
    elseif basis == 2
        basis1d_sym = basis_chebyshev(2d, x)
    elseif basis == 3
        basis1d_sym = basis_gegenbauer(2d, n, x)
    elseif basis == 4
        basis1d_sym = basis_laguerre(2d, BigFloat(n)/2-1,x)
        max_coef = [max(coefficients(basis1d_sym[i])...) for i = 1:length(basis1d_sym)]
        basis1d_sym = [max_coef[i]^(-1) * basis1d_sym[i] for i = 1:length(basis1d_sym)]
    else
        # also monomial basis for wrong input
        println("The option 'basis = $basis' was not recognized, we use the monomial basis as initial basis.")
        basis1d_sym = m(x, 2d)
    end
    if verbose
        println("Creating the polynomials for the SOS basis and the potential samples...")
    end
    # with the smart_eval basis, we don't compute the polynomials explicitely, but just the evaluations.
    sym_basis_smart_eval = [(u,v,t) -> basis1d_sym[deg-3k-2j+1](u+v+t) * basis1d_sym[j+1](u*v+v*t+u*t) * basis1d_sym[k+1](u*v*t) for deg=0:2d for k=0:div(deg,3) for j=0:div(deg-3k,2)]

    # we don't have a lot of double samples point due to symmetry, because we don't use exactly the same point sets for the x,y and z
    #We create a grid of sample points. either equally spaced or chebyshev-like
    if grid_option == 1
        samples = [-1 .+ (1+costheta) .* [i/N,j/N,k/N] for i=0:N for j=i:N for k=j:N] #this is kinda useless
    elseif grid_option == 2
        cheb_points = [vcat(sample_points_chebyshev(N+k,-1,costheta)...) for k=0:2]
        samples = [[cheb_points[1][i+1], cheb_points[2][j+1], cheb_points[3][k+1]] for i=0:N for j=0:N+1 for k=0:N+2]
    end
    samples = [
        s for s in samples if (1 + 2 * u * v * t - u^2 - v^2 - t^2)(s...) >= 0
    ] # We only keep the samples in the domain. This is not really needed
    #The degrees of the polynomials. They are ordered on degree, so we can look for the last one to determine the length of the bases we need later.
    degrees = [deg for deg=0:2d for k=0:div(deg,3) for j=0:div(deg-3k,2)]
    last_degree = [findlast(x -> x == k, degrees) for k=0:2d]

    if verbose
        println("Computing the approximate Fekete points and a corresponding basis...")
    end
    sym_basis, samples = approximatefekete(sym_basis_smart_eval,samples)

    if verbose
        println("Creating the SOS part of the trivariate constraint...")
    end
    for wi=1:length(weights)
        for swi=1:length(symmetry_weights)
            rank = length(symmetry_weights[swi])
            #this has in general too many entries, so we remove the ones with too high degree
            #kron(a,b) does [a[1]*b, a[2]*b,..., a[end] * b]. So the degree of vecs[r][i] is total_degree(symmetry_weights[swi][r][div(i,length(b))]+degree[(i-1)%length(b)+1]
            vecs = [kron(symmetry_weights[swi][r], sym_basis[1:last_degree[1+2d-total_degree(weights[wi])]]) for r=1:rank]
            for r=1:rank
                len = last_degree[1+2d-total_degree(weights[wi])] # the length of the symmetric basis we tensored the symmetric weight with
                #We keep the entries if the resulting diagonal element has degree at most 2d-deg(weight), since the weight will be set as the 'eigenvalue'. Note that this allows us to use sample points outside the domain
                keep_idx = [i for i=1:length(vecs[r])
                    if 2*(total_degree(symmetry_weights[swi][r][div(i-1,len)+1]) + degrees[(i-1)%len+1]) <= 2d-total_degree(weights[wi])]
                vecs[r] = vecs[r][keep_idx]
            end
            F[Block((:trivariatesos, wi,swi))] = LowRankMatPol([weights[wi] for r=1:rank], vecs)
        end
    end
    trivariatecon = Constraint(0, F, Dict(), samples)

    if verbose
        println("Creating the univariate constraint")
    end
    W, (w,) = PolynomialRing(FF, ["w"])
    f = Dict()
    for k=0:d # F(u,u,1)
        # from the (normally) three vu^T items, we only need 2 if we sum the ones with common u^T
        f[Block((:F,k))] = LowRankMatPol([1,1],
                            [m(w,d-k), m(W(1),d-k)],
                            [Q(n-1,k,w,w,1) .* m(w,d-k)+Q(n-1,k,w,1,w) .* m(W(1),d-k),Q(n-1,k,1,w,w) .* m(w,d-k)])
    end
    basis = basis_gegenbauer(2d, n, w)
    for k=0:2d # a_k
        f[Block((:a,k))] = LowRankMatPol([basis[k+1]], [[1]])
    end
    uni_basis = basis_chebyshev(2d, x)
    samples1d = sample_points_chebyshev(2d, -1, costheta) #we can take more samples, now we just get a better basis based on these samples. But since chebyshev points are relatively good, this is not a problem
    uni_basis,samples1d = approximatefekete(uni_basis, samples1d)

    f[Block((:uni_SOS,1))] = LowRankMatPol([1], [uni_basis[1:d+1]])
    f[Block((:uni_SOS,2))] = LowRankMatPol([p(w,costheta)], [uni_basis[1:d]])

    univariatecon = Constraint(-1, f, Dict(), samples1d)

    obj = Dict()
    obj[Block((:F, 0))] = ones(d+1, d+1)
    for k=0:2d
        obj[Block((:a,k))] = ones(1, 1)
    end
    objective = Objective(1, obj, Dict())

    sos = LowRankPolProblem(false, objective, [univariatecon, trivariatecon])
    if verbose
        println("Converting to a clustered low-rank SDP...")
    end
    if all_free
        sdp = ClusteredLowRankSDP(sos,as_free=[(:F,k) for k=0:d])
    else
        sdp = ClusteredLowRankSDP(sos)
    end
    status, sol, time, errorcode = solvesdp(sdp;kwargs...)
end
export three_point_spherical_cap_highrank
function three_point_spherical_cap_highrank(n, d, costheta, prec=256, all_free=false; basis=1, grid_option=2, N=2d, verbose=false, kwargs...)
    #basis: 1 = monomial
    #       2 = chebyshev
    #       3 = gegenbauer
    # These are orthogonalized with respect to the sample points, so it shouldn't really matter
    setprecision(BigFloat,prec)
    FF = RealField

    R, (u,v,t) = PolynomialRing(FF, ["u","v","t"])
    if verbose
        println("Creating the polynomials for F_k...")
    end
    F = Dict()
    for k=0:d
        @assert Q(n-1,k,u,v,t) == Q(n-1,k,v,u,t)
        #Instead of having both e.g. m(v,d-k) * m(u,d-k)^T and m(t,d-k) * m(u,d-k)^T, we combine them
        mat = Q(n-1,k,u,v,t) .* (m(v,d-k) * transpose(m(u,d-k)) +m(u,d-k) * transpose(m(v,d-k)) )
        mat += Q(n-1,k,t,u,v) .* (m(t,d-k) * transpose(m(u,d-k)) + m(u,d-k) * transpose(m(t,d-k)))
        mat += Q(n-1,k,t,v,u) .* (m(t,d-k) * transpose(m(v,d-k)) + m(v,d-k) * transpose(m(t,d-k)))

        F[Block((:F,k))] = 1//6 * mat
    end
    #The vectors which construct the Pi_pi matrices. We have to tensor them with a symmetric basis for the E_pi matrices
    symmetry_weights = [[[R(1)]],
                        [[(u-v)*(v-t)*(t-u)]],
                        [[1/sqrt(FF(2))*(2u-v-t),1/sqrt(FF(2))*(2v*t-u*t-u*v)],
                            [sqrt(FF(3)/FF(2))*(v-t),sqrt(FF(3)/FF(2))*(u*t-u*v)]]]
    #The polynomial weights which describe the domain
    weights = [R(1),
        p(u,costheta)+p(v,costheta)+p(t,costheta),
        p(u,costheta)*p(v,costheta)+p(v,costheta)*p(t,costheta)+p(t,costheta)*p(u,costheta),
        p(u,costheta)*p(v,costheta)*p(t,costheta),
        2u*v*t + 1 - u^2 - v^2 - t^2]
    #The basis in 1 variable used for the symmetric basis
    R1, (x,) = PolynomialRing(FF,["x"])
    if basis == 1
        basis1d_sym = m(x, 2d)
    elseif basis == 2
        basis1d_sym = basis_chebyshev(2d, x)
    elseif basis == 3
        basis1d_sym = basis_gegenbauer(2d, n, x)
    elseif basis == 4
        basis1d_sym = basis_laguerre(2d, BigFloat(n)/2-1,x)
        max_coef = [max(coefficients(basis1d_sym[i])...) for i = 1:length(basis1d_sym)]
        basis1d_sym = [max_coef[i]^(-1) * basis1d_sym[i] for i = 1:length(basis1d_sym)]
    else
        # also monomial basis for wrong input
        println("The option 'basis = $basis' was not recognized, we use the monomial basis as initial basis.")
        basis1d_sym = m(x, 2d)
    end
    if verbose
        println("Creating the polynomials for the SOS basis and the potential samples...")
    end
    # with the smart_eval basis, we don't compute the polynomials explicitely, but just the evaluations.
    sym_basis_smart_eval = [(u,v,t) -> basis1d_sym[deg-3k-2j+1](u+v+t) * basis1d_sym[j+1](u*v+v*t+u*t) * basis1d_sym[k+1](u*v*t) for deg=0:2d for k=0:div(deg,3) for j=0:div(deg-3k,2)]

    # we don't have a lot of double samples point due to symmetry, because we don't use exactly the same point sets for the x,y and z
    #We create a grid of sample points. either equally spaced or chebyshev-like
    if grid_option == 1
        samples = [-1 .+ (1+costheta) .* [i/N,j/N,k/N] for i=0:N for j=i:N for k=j:N] #this is kinda useless
    elseif grid_option == 2
        cheb_points = [vcat(sample_points_chebyshev(N+k,-1,costheta)...) for k=0:2]
        samples = [[cheb_points[1][i+1], cheb_points[2][j+1], cheb_points[3][k+1]] for i=0:N for j=0:N+1 for k=0:N+2]
    end
    samples = [
        s for s in samples if (1 + 2 * u * v * t - u^2 - v^2 - t^2)(s...) >= 0
    ] # We only keep the samples in the domain. This is not really needed
    #The degrees of the polynomials. They are ordered on degree, so we can look for the last one to determine the length of the bases we need later.
    degrees = [deg for deg=0:2d for k=0:div(deg,3) for j=0:div(deg-3k,2)]
    last_degree = [findlast(x -> x == k, degrees) for k=0:2d]

    if verbose
        println("Computing the approximate Fekete points and a corresponding basis...")
    end
    sym_basis, samples = approximatefekete(sym_basis_smart_eval,samples)

    if verbose
        println("Creating the SOS part of the trivariate constraint...")
    end
    for wi=1:length(weights)
        for swi=1:length(symmetry_weights)
            rank = length(symmetry_weights[swi])
            #this has in general too many entries, so we remove the ones with too high degree
            #kron(a,b) does [a[1]*b, a[2]*b,..., a[end] * b]. So the degree of vecs[r][i] is total_degree(symmetry_weights[swi][r][div(i,length(b))]+degree[(i-1)%length(b)+1]
            vecs = [kron(symmetry_weights[swi][r], sym_basis[1:last_degree[1+2d-total_degree(weights[wi])]]) for r=1:rank]
            for r=1:rank
                len = last_degree[1+2d-total_degree(weights[wi])] # the length of the symmetric basis we tensored the symmetric weight with
                #We keep the entries if the resulting diagonal element has degree at most 2d-deg(weight), since the weight will be set as the 'eigenvalue'. Note that this allows us to use sample points outside the domain
                keep_idx = [i for i=1:length(vecs[r])
                    if 2*(total_degree(symmetry_weights[swi][r][div(i-1,len)+1]) + degrees[(i-1)%len+1]) <= 2d-total_degree(weights[wi])]
                vecs[r] = vecs[r][keep_idx]
            end
            F[Block((:trivariatesos, wi,swi))] = LowRankMatPol([weights[wi] for r=1:rank], vecs)
        end
    end
    trivariatecon = Constraint(0, F, Dict(), samples)

    if verbose
        println("Creating the univariate constraint")
    end
    W, (w,) = PolynomialRing(FF, ["w"])
    f = Dict()
    for k=0:d # F(u,u,1)
        mat = Q(n-1,k,w,1,w) .* ( m(w,d-k) * transpose(m(1,d-k)) + m(1,d-k) * transpose(m(w,d-k)))
        mat += Q(n-1,k,w,w,1) .* ( m(w,d-k) * transpose(m(w,d-k)))

        # from the (normally) three vu^T items, we only need 2 if we sum the ones with common u^T
        f[Block((:F,k))] =  mat #this is 3 times the symmetrized  function
        # LowRankMatPol([1,1],
        #                     [m(w,d-k), m(W(1),d-k)],
        #                     [Q(n-1,k,w,w,1) .* m(w,d-k)+Q(n-1,k,w,1,w) .* m(W(1),d-k),Q(n-1,k,1,w,w) .* m(w,d-k)])
    end
    basis = basis_gegenbauer(2d, n, w)
    for k=0:2d # a_k
        f[Block((:a,k))] = hcat([basis[k+1]])#LowRankMatPol([basis[k+1]], [[1]])
    end
    uni_basis = basis_chebyshev(2d, x)
    samples1d = sample_points_chebyshev(2d, -1, costheta) #we can take more samples, now we just get a better basis based on these samples. But since chebyshev points are relatively good, this is not a problem
    uni_basis,samples1d = approximatefekete(uni_basis, samples1d)

    f[Block((:uni_SOS,1))] = LowRankMatPol([1], [uni_basis[1:d+1]])
    f[Block((:uni_SOS,2))] = LowRankMatPol([p(w,costheta)], [uni_basis[1:d]])

    univariatecon = Constraint(-1, f, Dict(), samples1d)

    obj = Dict()
    obj[Block((:F, 0))] = ones(d+1, d+1)
    for k=0:2d
        obj[Block((:a,k))] = ones(1, 1)
    end
    objective = Objective(1, obj, Dict())

    sos = LowRankPolProblem(false, objective, [univariatecon, trivariatecon])
    if verbose
        println("Converting to a clustered low-rank SDP...")
    end
    if all_free
        sdp = ClusteredLowRankSDP(sos,as_free=[(:F,k) for k=0:d])
    else
        sdp = ClusteredLowRankSDP(sos)
    end
    status, sol, time, errorcode = solvesdp(sdp;kwargs...)
end

end # of module
