# functions for pre and post processing of solutions
# e.g. removing linear dependent constraints, removing linearly dependent variables (and replacing them by 0 to get a solution to the original problem)

function find_linear_dependencies(sdp::ClusteredLowRankSDP; T=BigFloat, tol=sqrt(eps(T)))
    # finding the subblock sizes to get a total length of the constraint
    subblocksizes = [[0 for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            for r=1:size(sdp.A[j][l],1),s=1:size(sdp.A[j][l],2)
                for p in keys(sdp.A[j][l][r,s])
                    if subblocksizes[j][l] == 0
                        subblocksizes[j][l] = size(sdp.A[j][l][r,s][p],1)
                    elseif subblocksizes[j][l] != size(sdp.A[j][l][r,s][p],1)
                        s1 = subblocksizes[j][l]
                        s2 = size(sdp.A[j][l][r,s][p],1)
                        error("The subblocks (j,l,(r,s)) must have the same size for every r,s. $s1 != $s2")
                    end
                end
            end
        end
    end
    # 1) create matrix of linear system without the free variables in vars
    # 2) get linear dependencies through QR factorizations. 
    # for the constraints we only have to know which ones are linear dependent in the constraint matrices (for PSD vars)
    # but we have to use that to check whether we get linear constraints on the free variables
    # for the variables, we have to know the actual relations
    mfv = T.(vcat(sdp.B...))
    mpsd = hcat([vectorize_constraint(sdp, (j,p), subblocksizes; T) for j in eachindex(sdp.B) for p in axes(sdp.B[j], 1)]...) # columns are constraints
    cs_idx = [(j,p) for j in eachindex(sdp.B) for p in axes(sdp.B[j], 1)]
    rhs = T.(vcat(sdp.c...))

    # for constraints, we only need to find out which ones are linearly dependent in the PSD variables
    # those will either give completely linearly dependent constraints, or give linear constraints 
    # between the free variables (which we find in the next step) 
    Q, R, p = qr(mpsd, ColumnNorm()) # columns are constraints, and p is a permutation of the columns
    # size R: min(number of psd entries (indep), number of constraints) x number of constraints
    # linearly dependent constraints if some diagonal entries of R are (approximately) zero
    istart = findfirst(i->abs(R[i,i]) < tol, 1:minimum(size(R)))
    if size(mpsd, 1) < size(mpsd, 2) && isnothing(istart)
        # R is of the form [R1, R2] with R1 upper triangular with positive diagonal
        # so istart = nothing, but we still have linearly dependent constraints
        istart = size(mpsd, 1) + 1 
    end
        
    if !isnothing(istart) 
        cs = p[istart:end] # constraints that are removed
        # decomposition of the constraints in the linear independent constraints:
        # for the free variables we need to use this to find linear dependent variables
        # column i gives the decomposition of constraint istart+i-1 in terms of constraints p[1:istart-1] (according to the psd variables)
        # so we have the constraint i minus sum_j R[i, j] * constraint j is zero (or only free vars remaining)
        Rp = R[1:istart-1, 1:istart-1] \ R[1:istart-1, istart:end] 
        # size Rp: number of independent constraints x number of dependent constraints
    else # no linear dependencies
        cs = Int[]
        Rp = zeros(T, size(R,1), 0)
    end
    keepconstraints = [i for i=1:size(mfv,1) if !(i in cs)]
    

    # constraints on free variables through the lin dep constraints:
    freevar_mat = hcat(mfv, rhs) # size: number of constraints x (number of free variables + 1)
    # linear combinations of the constraints so that the PSD part equals 0:
    free_constraints = hcat(transpose(Rp), Matrix{T}(-I, size(Rp,2), size(Rp, 2))) * freevar_mat[p, :] 
    # get nice linear relations by isolating some variables (basically RREF, but using the QR factorization)
    Q, R, p = qr(free_constraints[:, 1:end-1], ColumnNorm()) # we don't want to take the 'constant' as a pivot
    if length(R) == 0 # for Float64, p = [0,..., 0] if R is of size 0xn. For BigFloat it works fine
        p = 1:size(R, 2)
    end
    pinv = [findfirst(==(i), p) for i in eachindex(p)]
    # size: number of constraints on the free variables (num lin dep constraints) x number of lin indep free variables
    rhs_changed = Q \ free_constraints[:, end] 
    # first 'free' variable index
    # the variables before this are determined by the constraints, given values of the variables after this
    istart = findfirst(i->abs(R[i,i]) < tol, 1:minimum(size(R))) 
    if isnothing(istart)
        # R is of the form [R1, R2] with R1 upper triangular with positive diagonal
        # so istart = nothing, but we still have linearly dependent variables
        if size(R, 1) < size(R, 2)
            istart = size(R, 1) + 1 
        elseif size(R, 2) > 0 #size(R, 1) >= size(R, 2); all variables are determined by the constraints
            istart = size(R, 2)+1
        end
        # if istart == nothing (still), we have size(R, 1) >= size(R,2) = 0, so no free variables in the problem
    end
    #TODO: check whether this goes right if R is a different size than mfv, or that we need a different condition to check this
    if !isnothing(istart) && istart <= length(rhs_changed)
        # rhs_changed = rhs_changed[1:istart-1]
        if !all(x->(abs(x) < tol), rhs_changed[istart:end]) 
            # some 'zero' constraint has nonzero rhs, so infeasible SDP
            error("Linear dependent constraint(s) resulting in a constraint 0 = b_i with b_i nonzero.")
        end
    end
    if !isnothing(istart) && istart > 1 # if istart = 1, R is the zero matrix, which doesn't give anything
        # 'free' free variables and variables that are determined by ff_vars and therefore removed from the sdp
        ff_vars = p[istart:end]
        nf_vars = p[1:istart-1]
        # for the free variables we need to use this to find linear dependent variables
        # column i gives the decomposition of constraint istart+i-1 in terms of constraints p[1:istart-1] (according to the psd variables)
        # so we have the constraint i minus R[i, j] * constraint j is zero (or only free vars remaining)
        Rref = R[1:istart-1, 1:istart-1] \ R[1:istart-1, istart:end]  # zero rows correspond to completely empty constraints
        rhs_changed = R[1:istart-1, 1:istart-1] \ rhs_changed[1:istart-1]
        # size: nf_vars x ff_vars (dependent x independent variables)
    else # no constraints
        Rref = zeros(T, 0, length(p))
        nf_vars = Int[]
        ff_vars = p
        rhs_changed = T[] # [ I Rref] x  = rhs_changed
    end
    
    #do the change and then remove linearly dependent variables
    changemat = vcat(-Rref, Matrix{T}(I, length(ff_vars), length(ff_vars)))[pinv, :] #variables end up in the order of ff_vars
    mfv_new = mfv[keepconstraints, :] * changemat
    Q2, R2, p2 = qr(mfv_new, ColumnNorm()) # to find free vars that perform the same function
    if length(R2) == 0 # for Float64, p = [0,..., 0] if R is of size 0xn. For BigFloat it works fine
        p2 = 1:size(R2, 2)
    end
    # # size mfv: number of constraints x number of free variables
    istart2 = findfirst(i->abs(R2[i,i]) < tol, 1:minimum(size(R2)))
    if size(mfv_new, 1) < size(mfv_new, 2) && isnothing(istart2)
        # R is of the form [R1, R2] with R1 upper triangular with positive diagonal
        # so istart = nothing, but we still have linearly dependent variables
        istart2 = size(mfv_new, 1) + 1 
    end
    if !isnothing(istart2)
        fv_zeros = p2[istart2:end] # we can set these free vars to 0 wlog
        fv_nonzeros = [i for i in 1:size(mfv_new, 2) if !(i in fv_zeros)]
    else
        fv_zeros = Int[]
        fv_nonzeros = 1:size(mfv_new,2)
    end
    total_removed = length(fv_zeros) + length(nf_vars)
    return [(i, cs_idx[i][1], cs_idx[i][2]) for i in cs], (fv_zeros, fv_nonzeros, Rref, rhs_changed, nf_vars, ff_vars), total_removed
end

function vectorize_constraint(sdp::ClusteredLowRankSDP, cidx, subblocksizes; T=BigFloat)
    # cidx = (j,p), where p is the constraintindex within the cluster
    jc, pc = cidx
    total_length = sum(binomial(size(sdp.A[j][l], 1) * subblocksizes[j][l]+1, 2) for j in eachindex(sdp.A) for l in eachindex(sdp.A[j]))
    v = zeros(T, total_length)
    k = 0
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            for r=1:size(sdp.A[j][l],1), s=1:r
                if r != s #off-diagonal block; need sum of (r,s) and (s,r) block
                    if j == jc # possibly nonzero
                        mrs = Matrix(get(sdp.A[j][l][r,s], pc, zeros(T, subblocksizes[j][l], subblocksizes[j][l])))
                        msr = Matrix(get(sdp.A[j][l][s,r], pc, zeros(T, subblocksizes[j][l], subblocksizes[j][l])))
                        mtot = msr+mrs
                        for i in eachindex(mtot)
                            v[k+i] = mtot[i]
                        end
                    end
                    k += subblocksizes[j][l]^2
                else
                    # r == s, so from matrix we take m[i1,i2] + m[i2,i1] for off-diagonal blocks
                    if j == jc
                        mrr = Matrix(get(sdp.A[j][l][r,s], pc, zeros(T, subblocksizes[j][l], subblocksizes[j][l])))
                        for i1=1:size(mrr, 1)
                            for i2 = 1:i1
                                if i1 == i2
                                    k+=1
                                    v[k] = mrr[i1,i2]
                                else
                                    k+=1
                                    v[k] = mrr[i1,i2] + mrr[i2, i1]
                                end
                            end
                        end
                    else
                        k += binomial(subblocksizes[j][l]+1, 2)
                    end
                end
            end
        end
    end
    return v
end


function remove_lindep_constraints!(sdp::ClusteredLowRankSDP, cs)
    # remove the linear dependent constraint in cs
    # cs are indices of the form (overal_constraint_index, clusterindex, constraint_index_incluster)
    # overal_constraint_index is not relevant here
    
    # first combine all the things in the same j, so that per j we only have to delete stuff once
    cs_j = Dict{Int, Vector{Int}}()
    for (i, j, p) in cs 
        if !haskey(cs_j, j)
            cs_j[j] = [p]
        else
            push!(cs_j[j], p)
        end
    end
    # per j, delete the constraints in one go
    for (j, ps) in cs_j
        non_ps = [i for i=1:size(sdp.B[j], 1) if !(i in ps)]
        sdp.B[j] = sdp.B[j][non_ps, :]
        sdp.c[j] = sdp.c[j][non_ps, :]
        for l in eachindex(sdp.A[j])
            for bl in eachindex(sdp.A[j][l])
                for p in ps
                    delete!(sdp.A[j][l][bl], p)
                    #TODO: we can consider if renaming the constraints is better or worse than using an additional dict for indexing in the solver
                end 
            end
        end
    end
    return sdp
end

function remove_lindep_freevars!(sdp::ClusteredLowRankSDP,  (fv_zeros, fv_nonzeros, Rref, rhs_changed, nf_vars, ff_vars); T=BigFloat)
    # remove the linear dependent free variables using the relations in var_rels
    # freevars in sdp: sdp.B with columns corresponding to variables, sdp.b with entries ('rows') corresponding to variables
    # we also might need to change the constant of the SDP, if var_rels replaces variables with something like a - sum_i a_i y_i
    # best would be to get some matrix that changes the variables correctly (since it is basically a substitution that should work)
    # then everything is just a matrix multiplication 
    if isempty(fv_zeros) && isempty(nf_vars) # no free variables removed
        return sdp
    end
    prec = precision(sdp.constant)
    pinv = [findfirst(==(i), vcat(nf_vars, ff_vars)) for i=1:length(nf_vars)+length(ff_vars)]
    changemat = vcat(-Rref, Matrix{T}(I, length(ff_vars), length(ff_vars)))[pinv, :] #variables end up in the order of ff_vars
    rhs_changed_ext = vcat(rhs_changed, zeros(BigFloat, length(ff_vars)))[pinv]
    #TODO change to Arblib.jl stuff where possible. 
    for j in eachindex(sdp.B)
        AL.sub!(sdp.c[j], sdp.c[j], ArbRefMatrix(sdp.B[j][:, nf_vars] * rhs_changed_ext[nf_vars]); prec)
        sdp.B[j] = ArbRefMatrix(sdp.B[j] * changemat[:, fv_nonzeros]; prec)
    end
    AL.add!(sdp.constant, sdp.constant,  dot(sdp.b[nf_vars], rhs_changed))
    sdp.b = ArbRefMatrix(transpose(changemat[:, fv_nonzeros]) * sdp.b; prec)
    return sdp
end

function add_zeros_constraintdual(x, cs; T=BigFloat)
    # after solving, we have to add a 0 in the dual solution for every constraint that was removed
    # (since it is not used in the dual solution)
    csi = sort([t[1] for t in cs])
    xnew = T[]
    k = 1
    j = 1
    for i=1:length(x) + length(cs)
        if length(csi) >= j && csi[j] == i
            push!(xnew, T(0))
            j+=1
        else
            push!(xnew, T(x[k]))
            k+=1
        end
    end
    return xnew
end



function add_dependent_freevars(y, (fv_zeros, fv_nonzeros, Rref, rhs_changed, nf_vars, ff_vars); T=BigFloat)
    # add the linear dependent free variables to the solution
    # 1) add zeros for fv_zeros
    # 2) use Rref, rhs_changed, nf_vars, ff_vars to determine the linearly dependent variables from the indep ones
    # we have [I Rref][y_nf, y_f] = rhs, so y_nf = rhs - Rref * y_f
    ynew = []
    k = 1
    for i=1:length(fv_zeros)+length(fv_nonzeros)
        if i in fv_zeros
            push!(ynew, T(0))
        else
            push!(ynew, T(y[k]))
            k += 1 
        end
    end
    nf_varspart = rhs_changed - Rref * ynew
    pinv = [findfirst(==(i), vcat(nf_vars, ff_vars)) for i=1:length(nf_vars)+length(ff_vars)]
    return vcat(nf_varspart, ynew)[pinv]
end

function preprocess!(sdp::ClusteredLowRankSDP; T=BigFloat, tol=sqrt(eps(T)))
    # we need BigFloat because qr factorization is not available in Arblib.jl
    # for correctness, we need the same precision
    bfprec = precision(T)
    if bfprec < precision(sdp.constant) && T==BigFloat
        setprecision(T, precision(sdp.constant))
    end
    # first test whether we need anything removed, if so, calculate the correct changes in high precision
    # this saves a lot of time for big SDPs without linear dependencies
    cs, var_rels, total_free_removed = find_linear_dependencies(sdp; T=Float64)
    if length(cs) > 0 || total_free_removed > 0
        cs, var_rels, total_free_removed = find_linear_dependencies(sdp; T, tol)
    else
        return cs, var_rels # empty, so no changes
    end

    remove_lindep_constraints!(sdp, cs)
    remove_lindep_freevars!(sdp, var_rels; T)
    # give warnings when constraints/variables are found to be dependent
    if length(cs) > 0
        @warn "$(length(cs)) constraints were removed due to linear dependencies."
    end
    if total_free_removed > 0
        @warn "$total_free_removed free variables were removed due to linear relations or dependencies."
    end
    # set precision of BigFloat back to what it was
    if T==BigFloat
        setprecision(BigFloat, bfprec)
    end
    return cs, var_rels
end

function postprocess(x, y, cs, var_rels; T=BigFloat)
    # use the precision of the variables
    bfprec = precision(T)
    # x corresponds to constraints, so x has length >= 1
    if bfprec < precision(first(x)) && T==BigFloat
        setprecision(T, precision(first(x)))
    end
    x = add_zeros_constraintdual(x, cs; T)
    y = add_dependent_freevars(y, var_rels; T)
    if T==BigFloat
        setprecision(T, bfprec)
    end
    return x, y
end


