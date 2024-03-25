"""
    basis_monomial(d::Int, x...)

Generate the monomial basis in variables x... up to degree d (inclusive).
"""
function basis_monomial(d::Int, x...)
    n = length(x)
    if n > 1
        q = Vector{typeof(prod(one.(x)))}(undef, binomial(n+d,d))
    elseif n == 1
        q = Vector{typeof(one(x...))}(undef, binomial(n+d,d))
    end
    idx = 1
    for k=0:d
        for exponent in multiexponents(n,k)
            q[idx] = prod([x[i]^exponent[i] for i in eachindex(exponent)])
            idx+=1
        end
    end
    return q
end

"""
    basis_laguerre(d::Integer, alpha, x)

Generate the [Laguerre polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials) with parameter alpha up to degree d (inclusive). 
"""
function basis_laguerre(d::Integer, alpha, x)
    v = Vector{typeof(one(alpha) * one(x))}(undef, 1 + d)
    v[1] = one(x)
    d == 0 && return v
    v[2] = 1 + alpha - x
    d == 1 && return v
    for l = 2:d
        v[l+1] = 1 // big(l) * ((2l - 1 + alpha - x) * v[l] - (l + alpha - 1) * v[l-1])
    end
    return v
end

"""
    basis_jacobi(d::Integer, alpha, beta, x, normalized = true)

Generate the [Jacobi polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials) with parameters alpha and beta up to degree d (inclusive). 
"""
function basis_jacobi(d::Integer, alpha, beta, x, normalized = true)
    q = Vector{typeof(one(alpha) * one(x))}(undef, d + 1)
    q[1] = one(x)
    d == 0 && return q
    q[2] = x # normalized
    if !normalized
        q[2] *= (alpha + 1)
    end
    d == 1 && return q
    for k = 2:d
        # what if alpha+beta = -n for some integer 1 <= n < 2d ?
        q[k+1] =
            (2 * k + alpha + beta - 1) /
            BigFloat(2k * (k + alpha + beta) * (2k + alpha + beta - 2)) *
            ((2 * k + alpha + beta) * (2k + alpha + beta - 2) * x + beta^2 - alpha^2) *
            q[k] + -2 * (k + alpha - 1) * (k + beta - 1) * (2 * k + alpha + beta) * q[k-1]
    end
    return q
end

"""
    basis_chebyshev(d::Int,x)

Generate a basis of [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) up to degree d (inclusive). 
"""
function basis_chebyshev(d::Int,x)
    v = Vector{typeof(one(x))}(undef, 1 + d)
    v[1] = one(x)
    d == 0 && return v
    v[2] = x
    d == 1 && return v
    for l = 2:d
        v[l+1] = 2x*v[l] - v[l-1]
    end
    return v
end


"""
    basis_gegenbauer(d, n, x)

Basis for the Gegenbauer polynomials in dimension n up to degree d.
 This is the Gegenbauer polynomial with parameter lambda = n/2-1,
 or the Jacobi polynomial with alpha = beta = (n-3)/2.
 Normalized to evaluate to 1 at 1.
 Taken from arxiv/2001.00256, ancillary files, SemidefiniteProgramming.jl.
"""
function basis_gegenbauer(d, n, x)
    v = Vector{typeof(one(x))}(undef, 1 + d)
    v[1] = one(x)
    d == 0 && return v
    v[2] = x
    d == 1 && return v
    for l = 2:d
        v[l+1] = (2l + n - 4) // (l + n - 3) * x * v[l] - (l - 1) // (l + n - 3) * v[l-1]
    end
    return v
end

"""
    sample_points_simplex(n, d; T=BigFloat) -> Vector{Vector{T}}

Generate the rational sample points in the unit simplex with denominator d.
"""
function sample_points_simplex(n, d; T=BigFloat)
    #rational points in the unit simplex with denominator d
    #probably not very efficient, but I dont know how to do it better for general n.
    x = [zeros(T, n) for i = 1:binomial(n + d, d)] #need n+d choose d points for a unisolvent set, if symmetry is not used.
    idx = 1
    for I in CartesianIndices(ntuple(k -> 0:d, n)) #all tuples with elements in 0:d of length n
        if sum(Tuple(I)) <= d #in unit simplex
            x[idx] = [i / T(d) for i in Tuple(I)]
            idx += 1
        end
    end
    return x
end

"""
    sample_points_padua(d; T=BigFloat) -> Vector{Vector{T}}

Generate the [Padua points](https://en.wikipedia.org/wiki/Padua_points) for degree d.
"""
function sample_points_padua(d; T=BigFloat)
    #padua points:
    z = [Array{T}(undef, 2) for i = 1:binomial(2 + d, d)]
    z_idx = 1
    for j = 0:d
        delta_j = j % 2 == d % 2 == 1 ? 1 : 0
        mu_j = cospi(j / T(d))
        for k = 1:(div(d, 2)+1+delta_j)
            eta_k = j % 2 == 1 ? cospi((2 * k - 2) / T(d + 1)) : cospi((2 * k - 1) / T(d + 1))
            z[z_idx] = [mu_j, eta_k]
            z_idx += 1
        end
    end
    return z
end

"""
    sample_points_rescaled_laguerre(d; T=BigFloat) -> Vector{T}

Generate 'rescaled laguerre' points, as in SDPB.
"""
function sample_points_rescaled_laguerre(d; T=BigFloat)
    #as done in sdpb: ('rescaled Laguerre')
    # x[k] = sqrt(pi)/(64*(d+1)*log( 3- 2*sqrt(2))) * (-1+4*k)^2, with k=0:d
    constant = -sqrt(T(pi)) / (64 * (d + 1) * log(3 - 2 * sqrt(T(2))))
    x = zeros(T, d + 1)
    for k = 0:d
        x[k+1] = constant * (-1 + 4 * k)^2
    end
    return x
end

"""
    sample_points_chebyshev(d, a = -1, b = 1; T=BigFloat) -> Vector{T}

Generate the d+1 [Chebyshev points](https://en.wikipedia.org/wiki/Chebyshev_nodes) in [a,b].
"""
function sample_points_chebyshev(d, a = -1, b = 1; T=BigFloat)
    #roots of chebyshev polynomials of the first kind, unisolvent for polynomials up to degree d
    [
        (a + b) / T(2) +
        (b - a) / T(2) * cospi((2k - 1) / T(2(d + 1))) for
        k = 1:d+1
    ]
end

"""
    sample_points_chebyshev_mod(d, a = -1, b = 1; T=BigFloat) -> Vector{T}

Generate the d+1 modified chebyshev points in [a,b], the chebyshev points divided by cos(pi/(2(d+1))).
"""
function sample_points_chebyshev_mod(d, a = -1, b = 1; T = BigFloat)
    #roots of chebyshev polynomials of the first kind, divided by cos(pi/2(d+1)) to get a lower lebesgue constant
    [
        (a + b) / T(2) +
        (b - a) / T(2) * cospi((2k - 1) / T(2(d + 1))) /
        cospi(T(1) / 2(d + 1)) for k = 1:d+1
    ]
end
