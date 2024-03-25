if !isdefined(@__MODULE__, :ColumnNorm)
    # Make ColumnNorm work on old julia versions, it is introduced in Julia 1.7
    ColumnNorm() = Val(true)
end

function approximate_fekete(initial_points, basis;prec=precision(BigFloat), verbose=false, show_det = false, s = 3,alg=[:BigFloat, :Arb, :high_prec][2])
    # P contains the basis transformation from the monomial basis to the new basis. 
    # the returned V,P are BigFloats. Otherwise it doesn't work with for example ArbField
    # This destroys the use of error bounds though.
    verbose && print("Evaluating polynomials... ")
    t = @elapsed V = [pol(point...) for point in initial_points, pol in basis]
    verbose && println(t)
    if alg == :BigFloat
        V, P, point_indices = approximate_fekete_bigfloat(V, s=s, show_det=show_det, prec=prec, verbose=verbose)
    elseif alg == :Arb
        V, P, point_indices = approximate_fekete_arb(V, s=s, show_det=show_det, prec=prec, verbose=verbose)
    elseif alg == :high_prec
        V, P, point_indices = approximate_fekete_highprec(V, s=s, show_det=show_det, prec=prec)
    end
    p = sortperm(initial_points[point_indices])
    return V[p,:], P, initial_points[point_indices][p]
end

function approximate_fekete_bigfloat(V; s = 3, verbose=false, show_det = false,prec=precision(BigFloat))
    setprecision(BigFloat, prec) do

        # we do the QR factorizations in floating point precision, and the basis changes in high precision (BigFloats)
        P = Matrix{BigFloat}(I, size(V, 2), size(V, 2)) # to keep track of the basis change
        for k = 1:s
            verbose && println("QR iteration $k / $s")
            F = qr(Float64.(V))
            U = BigFloat.(I / F.R)
            V = V * U # should be approximately equal to Q
            P = P * U #keep track of the basis change
        end
        # Find the good points:
        F2 = qr(Float64.(V'), ColumnNorm()) # this is deprecated in 1.7, but the new thing (ColumnNorm()) doesn't work in 1.6
        point_indices = F2.p[1:size(V,2)]
        # Do a last basis change to get a good basis for these points
        F = qr(Float64.(V[point_indices,:]))
        U = BigFloat.(I / F.R)
        V = V[point_indices,:] * U # the new V is square
        P = P * U
        if show_det
            println(det(V))
        end
        return V, P, point_indices
    end
end

function approximate_fekete_arb(V::Matrix{T}; verbose=false, s = 3,show_det = false,prec=precision(BigFloat)) where T
    setprecision(BigFloat, prec) do
        # We do the QR factorizations in floating point precision, and the basis changes in high precision (Arb)
        V = AL.ArbMatrix(BigFloat.(V),prec=prec)
        P = AL.ArbMatrix(size(V, 2), size(V, 2),prec=prec) # to keep track of the basis change
        Arblib.one!(P)
        for k = 1:s
            verbose && println("QR iteration $k / $s")
            F = qr(Float64.(V))
            U = AL.ArbMatrix(I / F.R,prec=prec)
            Arblib.approx_mul!(V,V,U)
            Arblib.approx_mul!(P,P,U)
        end
        # Find the good points:
        F2 = qr(Float64.(V'), ColumnNorm())
        point_indices = F2.p[1:size(V,2)]
        # Do a last basis change to get a good basis for these points
        V = V[point_indices,:]
        F = qr(Float64.(V))
        U = AL.ArbMatrix(I / F.R,prec=prec)
        Arblib.approx_mul!(V,V,U)
        Arblib.approx_mul!(P,P,U)
        if show_det
            det_V = Arb(prec=prec)
            Arblib.det!(det_V,V)
            println(det_V)
        end
        return BigFloat.(V), BigFloat.(P), point_indices
    end
end

function approximate_fekete_highprec(V::Matrix{T}; s = 3,show_det = false, prec=precision(BigFloat)) where T
    setprecision(BigFloat, prec) do

        # We do one basis change, and for that we compute the qr factorization in high precision with BigFloats
        # Then we do the basis change with Arb for speed. Note that especially the QR factorization can take long for large matrices (say 200+ rows/cols)
        # Find the basis change matrix (from the QR factorization)
        F = qr(BigFloat.(V))
        U = AL.ArbMatrix(F.R,prec=prec)
        Arblib.approx_inv!(U,U)

        #Do the basis change for V
        V = AL.ArbMatrix(V,prec=prec)
        Arblib.approx_mul!(V,V,U)

        # Do a QR factorization of V^T to get a subset of good sample points
        F2 = qr(BigFloat.(V'), ColumnNorm())
        point_indices = F2.p[1:size(V,2)]
        # Do a last basis change to get a good basis for these points
        V = V[point_indices,:]

        F = qr(BigFloat.(V))
        U2 = AL.ArbMatrix(F.R,prec=prec)
        Arblib.approx_inv!(U2,U2)
        Arblib.approx_mul!(V,V,U2)
        Arblib.approx_mul!(U,U,U2)

        if show_det
            det_V = Arb(prec=prec)
            Arblib.det!(det_V,V)
            println(det_V)
        end
        return V, U, point_indices
    end
end

# use an exact basis transformation
"""
    approximatefeketeexact(R, basis, samples)

Apply approximate fekete but return an exact basis transformation.
"""
function approximatefeketeexact(R, basis, samples; s=3)
    esamples = [R.(rationalize.(BigInt, Float64.(sample), tol=1e-3)) for sample in samples]

	print("Evaluating polynomials in exact arithmetic...")
    t = @elapsed eV = [pol(esample...) for esample in esamples, pol in basis]
    println(" ($(t)s)")

	aV = BigFloat.(eV)

    P = Matrix{BigFloat}(I, size(eV, 2), size(eV, 2))
    for k = 1:s
        F = qr(Float64.(aV))
        U = BigFloat.(I / F.R)
        aV *= U # should be approximately equal to Q
        P *= U # keep track of the basis change
    end

    # Find the good points
    G = qr(aV', ColumnNorm())
    sample_indices = G.p[1:size(eV, 2)]
    # Do a last basis change to get a good basis for these points
    F = qr(Float64.(aV[sample_indices, :]))
    U = BigFloat.(I / F.R)
    P *= U

	eP = rationalize.(BigInt, P, tol=1e-3)

	# TODO verify basis transformation is invertible (in a better way)
	@assert !iszero(det(eP))

    print("Starting exact basis change...")
    t = @elapsed eV = eV[sample_indices, :] * eP # the new V is square
    println(" ($(t)s)")

    esamples = esamples[sample_indices]
    perm = sortperm(esamples)
    esamples = esamples[perm]
    eV = eV[perm, :]
    SR = SampledMPolyRing(R, esamples)
    [SampledMPolyRingElem(SR, eV[:, p]) for p in eachindex(basis)], esamples
end

# end #of module
