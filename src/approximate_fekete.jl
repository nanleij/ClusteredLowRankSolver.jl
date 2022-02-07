# module ApproximateFekete

using LinearAlgebra, GenericLinearAlgebra, Arblib
# export approximate_fekete

function approximate_fekete(initial_points, basis;show_det = false, s = 3,alg=[:BigFloat, :Arb, :high_prec][2])
    V = [pol(point...) for point in initial_points, pol in basis]
    if alg == :BigFloat
        V, P, point_indices = approximate_fekete_bigfloat(V,s=s,show_det=show_det)
    elseif alg == :Arb
        V, P, point_indices = approximate_fekete_arb(V,s=s,show_det=show_det)
    elseif alg == :high_prec
        V, P, point_indices = approximate_fekete_highprec(V,s=s,show_det=show_det)
    end
    return V, P, initial_points[point_indices]
end

function approximate_fekete_bigfloat(V; s = 3, show_det = false)
    # we do the QR factorizations in floating point precision, and the basis changes in high precision (BigFloats)
    P = Matrix{BigFloat}(I, size(V, 2), size(V, 2)) # to keep track of the basis change
    for k = 1:s
        F = qr(Float64.(V))
        U = BigFloat.(I / F.R)
        V = V * U # should be approximately equal to Q
        P = P * U #keep track of the basis change
    end
    # Find the good points:
    F2 = qr(Float64.(V'), ColumnNorm())
    point_indices = F2.p[1:size(V,2)]
    # Do a last basis change to get a good basis for these points
    F = qr(Float64.(V[point_indices,:]))
    U = BigFloat.(I / F.R)
    V = V[point_indices,:] * U #the new V is square
    P = P * U
    if show_det
        println(det(V))
    end
    return V, P, point_indices
end

function approximate_fekete_arb(V::Matrix{T}; s = 3,show_det = false) where T
    #we do the QR factorizations in floating point precision, and the basis changes in high precision (Arb)

    prec = precision(eltype(V))
    V = ArbMatrix(V,prec=prec)
    P = ArbMatrix(size(V, 2), size(V, 2),prec=prec) #to keep track of the basis change
    Arblib.one!(P)
    for k = 1:s
        F = qr(Float64.(V))
        U = ArbMatrix(I / F.R,prec=prec)
        Arblib.approx_mul!(V,V,U)
        Arblib.approx_mul!(P,P,U)
    end
    # Find the good points:
    F2 = qr(Float64.(V'),  ColumnNorm())
    point_indices = F2.p[1:size(V,2)]
    # Do a last basis change to get a good basis for these points
    V = V[point_indices,:]
    F = qr(Float64.(V))
    U = ArbMatrix(I / F.R,prec=prec)
    Arblib.approx_mul!(V,V,U)
    Arblib.approx_mul!(P,P,U)
    if show_det
        det_V = Arb(prec=prec)
        Arblib.det!(det_V,V)
        println(det_V)
    end
    return T.(V), T.(P), point_indices
end

function approximate_fekete_highprec(V::Matrix{T}; s = 3,show_det = false) where T
    #We do one basis change, and for that we compute the qr factorization in high precision with BigFloats
    #Then we do the basis change with Arb for speed. Note that especially the QR factorization can take long for large matrices (about >200 rows/cols)
    #find the basis change matrix (from the QR factorization)
    prec=precision(V[1,1])
    F = qr(BigFloat.(V))
    U = ArbMatrix(F.R,prec=prec)
    Arblib.approx_inv!(U,U)

    #Do the basis change for V
    V = ArbMatrix(V,prec=prec)
    Arblib.approx_mul!(V,V,U)

    # Do a QR factorization of V^T to get a subset of good sample points
    F2 = qr(BigFloat.(V'), ColumnNorm())
    point_indices = F2.p[1:size(V,2)]
    # Do a last basis change to get a good basis for these points
    V = V[point_indices,:]

    F = qr(BigFloat.(V))
    U2 = ArbMatrix(F.R,prec=prec)
    Arblib.approx_inv!(U2,U2)
    Arblib.approx_mul!(V,V,U2)
    Arblib.approx_mul!(U,U,U2)

    if show_det
        det_V = Arb(prec=prec)
        Arblib.det!(det_V,V)
        println(det_V)
    end
    return T.(V), T.(U), point_indices
end

# end #of module
