function select_vals(primalsol::PrimalSolution{T}, dualsol::DualSolution{T}, max_d=10; valbound=1e-15,errbound=1e-15, bits=100, max_coeff=1000, sizebound=10^6) where T
    all_vals = Tuple{T,Int64}[]
    for (k,m) in primalsol.matrixvars
        #if the primal solution has large entries, use the eigenvalues (svd) of the dual solution
        if maximum(abs.(m)) > sizebound  && maximum(abs.(matrixvar(dualsol, k))) < sizebound || true
            tmp = svd(matrixvar(dualsol, k))
            num_evs = count(x->abs(x)< valbound, tmp.S)
            if num_evs  == 0 
                continue
            end
            m = Matrix(transpose(tmp.U[:, end-num_evs+1:end]))
        end
        m = RowEchelon.rref!(copy(m), valbound)
        vecs = [m[i, :] for i in axes(m,1) if norm(m[i, :]) > valbound]
        if length(vecs) == 0
            @show "$k has no kernelvectors without 0-1 entries"
            continue
        end
        idxs = [findfirst(x->abs(x)>valbound, v[length(vecs)+1:end]) for v in vecs]
        vals = [vecs[i][length(vecs)+idxs[i]] for i in eachindex(vecs) if !isnothing(idxs[i])]
        for v in vals
            for d = 1:max_d
                coeffs = nothing
                try
                    coeffs = min_poly(v, d, bits=bits, errbound=errbound)
                catch e
                    continue
                end
                # Q: when can you say that a minimal polynomial is found?
                if all(abs(x) <= max_coeff for x in coeffs)
                    #probably found a good minimal polynomial
                    if d > 1 # we don't care about rationals
                        push!(all_vals, (v,d))
                    end
                    #found a nice polynomial of degree d, so we won't test for higher degree
                    break 
                end
            end
        end
    end
    return all_vals
end

function find_common_minpoly(generators; max_coeff=1000, bits=100, errbound=1e-15)
    if length(generators) == 0
        return QQ(1), 1, [-1,1], QQ 
    end
    # start with one with maximum degree, with minimum size of coefficients
    g,d = argmax(x->(x[2], -sum(abs.(min_poly(x..., bits=bits, errbound=errbound)))), generators)
    for i in eachindex(generators)
        v, degv = generators[i]
        if degv <= d #possibly decomposable
            switch=false
            coeffs = decompose(v, g, d, bits=bits, errbound=errbound)
        else
            switch=true
            coeffs = decompose(g, v, degv, bits=bits, errbound=errbound)
        end
        if all(abs(x) < max_coeff for x in coeffs)
            if switch
                # g can be decomposed in sum_i a_i v^i, so everything before too
                g,d = v, degv
            end
        else
            @info "Extending the field"
            # g, v do not live in the field generated by one of them
            g = g+v 
            #field has at most degree d+degv (probably exactly, but maybe not)
            for deg = max(d,degv):d+degv
                coeffs = min_poly(g, deg, bits=bits, errbound=errbound)
                if all(abs(x) < max_coeff for x in coeffs)
                    d = deg
                    break
                end
            end
        end
    end
    coeffs = min_poly(g,d, bits=bits, errbound=errbound)
    R, x = polynomial_ring(QQ, :x)
    N, z = number_field(sum(coeffs[i] * x^(i-1) for i=1:d+1), :z)
    return g, d, coeffs, N
end



"""
    find_field(primalsol, dualsol, max_degree=4; valbound=1e-15, errbound=1e-15, bits=100, max_coeff=1000)

Heuristically find a field over which the kernel can probably be defined. 

Only consider values at least `valbound` in absolute value. Find minimal polynomials 
such that the chosen entries are approximately generators with an error bound of `errbound`.
Use `bits` number of bits and reject minimal polynomials with a maximum coefficient of more than `max_coeff`.
"""
function find_field(primalsol::PrimalSolution{T}, dualsol::DualSolution{T}, max_degree=10; valbound=1e-15,errbound=1e-15, bits=max_degree*100, max_coeff=10^5) where T
    vals = select_vals(primalsol, dualsol, max_degree;valbound=valbound, errbound=errbound, bits=bits, max_coeff=max_coeff)
    g, d, coeffs, N = find_common_minpoly(vals; max_coeff=max_coeff, errbound=errbound, bits=bits)
    #if Hecke:
    # N, _ = simplify(N)
    if N == QQ
        return N, BigFloat(1)
    else
        g = BigFloat(root_balls(N, g, precision(BigFloat))[1]) #get a more precise root
    end
    return N, g
end

#Q: does the number of bits for which it possibly is nice grow with the degree?
function min_poly(g, d; bits=100, errbound=1e-15)
    clindep([g^k for k=0:d, j=1:1], bits, errbound)
end

function decompose(v, g, d; bits=100, errbound=1e-15)
    a = clindep(reshape([v, [g^k for k=0:d-1]...], d+1, 1), bits, errbound)
end

"""
    to_field(v, N, g; bits=100, errbound=1e-15)

Find an approximation of v in the number field N, using the approximate generator g of N.
"""
function to_field(v, N::AbsSimpleNumField, g; bits=100, errbound=1e-15)
    a = decompose(v, g, degree(N); bits=bits, errbound=errbound)
    z = gen(N)
    coeffs = -a[2:end] .// a[1]
    return sum(coeffs[i] * z^(i-1) for i=1:degree(N))
end
