""" 
    RoundingSettings(settings...)

Settings for the rounding procedure:
1. Finding the kernel
    - `kernel_errbound`: (default: `1e-10`) the allowed error for the kernel vectors. That is, the maximum entry of Xv is in absolute value at most this
    - `kernel_round_errbound`: (default: `1e-15`) the maximum allowed error when rounding the reduced row-echelon form entrywise
    - `kernel_use_dual`: (default: `true`) use the dual solution to find the kernel vectors  (otherwise, use an SVD of the primal solution)
    - `kernel_lll`: (default: `false`) if true, use the LLL algorithm to find the nullspace of the kernel. Otherwise, use the reduced row-echelon form to find kernel vectors.
    - `kernel_bits`: (default: `1000`) the maximum number of bits to be used in the LLL algorithm (for finding relations or finding the entries of the RREF)
2. Reducing the kernel vectors
    - `reduce_kernelvectors`: (default: `true`) apply the reduction step or not
    - `reduce_kernelvectors_cutoff`: (default: `400`) do reduction on the full matrix if its size is at most this cutoff. Otherwise do it on a submatrix
    - `reduce_kernelvectors_stepsize`: (default: `200`) the number of extra columns to take into account in each iteration of the reduction step
3. Transforming the problem and the solution
    - `unimodular_transform`: (default: `true`) use a unimodular transform obtained in the reduction step
    - `normalize_transformation`: (default: `true`) multiply by a diagonal matrix to get an integer transformation for the problem (for problems over QQ)
4. Finding an approximate solution in the field
    - `regularization`: (default: `1e-20`) use this regularization for solving the extended system
    - `approximation_decimals`: (default: `40`) Approximate the numerical solution using this many digits, entrywise
5. Rounding the solution to the affine space of constraints
    - `redundancyfactor`: (default: `10`) take at least this times the number of constraints columns as potential pivots
    - `pseudo`: (default: `true`) use the psuedo inverse for rounding (this may give solutions with larger bitsize than needed)
    - `pseudo_columnfactor`: (default: `1.05`) For a system of r rows, use r * pseudo_columnfactor number of columns for the pseudo inverse
    - `extracolumns_linindep`: (default: `false`) if true, take the extra columns linearly independent of each other (otherwise, random columns)
"""
struct RoundingSettings
    kernel_lll::Bool # use LLL to find kernel vectors (vs rref)
    kernel_bits::Int64
    kernel_errbound::Float64
    kernel_round_errbound::Float64
    kernel_use_dual::Bool
    reduce_kernelvectors::Bool # use the reduction heuristic
    reduce_kernelvectors_cutoff::Int64
    reduce_kernelvectors_stepsize::Int64
    unimodular_transform::Bool
    approximation_decimals::Int64
    regularization::Float64
    normalize_transformation::Bool
    redundancyfactor::Int
    pseudo::Bool
    pseudo_columnfactor::Float64
    extracolumns_linindep::Bool
    function RoundingSettings(;kernel_lll=false,
                               kernel_bits=1000,
                               kernel_errbound=1e-10,
                               kernel_round_errbound=1e-15, #with 1e-20, 1e-40 eps is not enough for the larger matrices
                               kernel_use_dual=true,
                               reduce_kernelvectors=true,
                               reduce_kernelvectors_cutoff = 400, #only for very big matrices we want this
                               reduce_kernelvectors_stepsize = 200,
                               unimodular_transform=true,
                               approximation_decimals=40,
                               regularization=1e-20,
                               normalize_transformation=true,
                               redundancyfactor=10,
                               pseudo=true, 
                               pseudo_columnfactor=1.05, 
                               extracolumns_linindep=false)
        if pseudo_columnfactor < 1
            pseudo_columnfactor = 1
            @info "We take pseudo_columnfactor to be 1 (the minimum possible)"
        end
        new(kernel_lll,
            kernel_bits,
            kernel_errbound,
            kernel_round_errbound,
            kernel_use_dual,
            reduce_kernelvectors,
            reduce_kernelvectors_cutoff,
            reduce_kernelvectors_stepsize,
            unimodular_transform,
            approximation_decimals,
            regularization,
            normalize_transformation,
            redundancyfactor,
            pseudo,
            pseudo_columnfactor,
            extracolumns_linindep)
    end
end

function preprocess_rows!(A, b; include_b=false)
    for r in axes(A,1)
        l = lcm(Vector(denominator.(A[r, :])))
        if include_b
            l = lcm(l, denominator(b[r,1]))
        end
        A[r, :] *= l
        b[r, 1] *= l 
    end
    return A, b
end

function select_columns(problem::Problem, sol::PrimalSolution, redundancyfactor=6; verbose=true)
    nconstraints = sum(x->length(x.samples), problem.constraints)
    x = vectorize(sol)
    nvars = length(x)

    if redundancyfactor < 0
        return collect(1:nvars)
    end

    # Select the elements on the diagonal and directly above/below it 
    v = as_primal_solution(sol, zeros(Int,nvars))
    for (k, m) in v.matrixvars
        m = zero(m)
        for i in axes(m,1)
            m[i,i] = 1
            if size(m,2) >= i+1
                m[i,i+1] = m[i+1,i] = 1
            end
        end
        v.matrixvars[k] = m
    end
    # give the entries occurring in objective a 2; those might be more important
    for (k, m) in problem.objective.matrixcoeff
        nonzeroentries = m .!= 0
        for i in eachindex(nonzeroentries)
            if nonzeroentries[i]
                v.matrixvars[k][i] += 2
            end
        end
    end 
    for (k, m) in problem.objective.freecoeff
        if m != 0
            v.freevars[k] += 2
        end
    end 


    vvec = vectorize(v)
    obj_cols = [i for (i,val) in enumerate(vvec) if val >= 2]
    chosen_nonobjcols = [i for (i,val) in enumerate(vvec) if val == 1]
    if length(obj_cols) + length(chosen_nonobjcols) > redundancyfactor * nconstraints
        # we have a lot of entries already, so we  chose a random subset of them
        pivot_cols = shuffle!(vcat(obj_cols, chosen_nonobjcols))[1:redundancyfactor*nconstraints]
        #put the objective colums first, so they are more likely to be chosen as pivots
        pivot_cols = unique(vcat([i for i in pivot_cols if i in obj_cols], pivot_cols))
    else
        pivot_cols = vcat(obj_cols, chosen_nonobjcols)
    end

    nneeded = redundancyfactor * nconstraints - length(pivot_cols)
    nneeded = max((redundancyfactor-2) * nconstraints, nneeded)

    notchosen = shuffle!([i for (i,val) in enumerate(vvec) if val == 0])
	append!(pivot_cols, notchosen[1:min(nneeded, length(notchosen))])

	verbose && println("  Reducing the system from $(length(vvec)) columns to $(length(pivot_cols)) columns")
    
    return pivot_cols
end

function project_to_affine_space(problem::Problem, sol::PrimalSolution; FF=QQ, g=1, settings::RoundingSettings=RoundingSettings(), monomial_bases=nothing, verbose=true)
    finished = false
    extra_redundancy = 0
    while !finished
        column_selection = select_columns(problem, sol, settings.redundancyfactor + extra_redundancy, verbose=verbose)
        A, b, x, column_selection = get_rational_system(problem, sol, FF, g; columns = column_selection, monomial_bases=monomial_bases, settings=settings, verbose=verbose)
        # print_memory_info()
        x_extra, correct_slacks, finished = project_to_affine_space(A,b; settings=settings, verbose=verbose)
        if !finished && length(x) > length(column_selection)
            # take more columns in the next iteration, since the system is inconsistent
            extra_redundancy += 2
            continue
        elseif !finished && length(x) == length(column_selection)
            error("The system is inconsistent but we took all the columns")
        end
        if finished
            for (j,i) in enumerate(column_selection)
                x[i] += x_extra[j]
            end
            # we still need to go back to the field (what's a good name?)
            x_FF = xrational_to_field(x, FF)
            return as_primal_solution(sol, x_FF), correct_slacks
        end
    end
end


function project_to_affine_space(A::QQMatrix, b::QQMatrix; settings::RoundingSettings=RoundingSettings(), verbose=true)
    verbose && print("  Preprocessing to get an integer system...")
    t = @elapsed A, b = preprocess_rows!(A, b)
    verbose && println(" ($(t)s)")
    # print_memory_info()

    
    rows = 1:nrows(A)
    verbose && print("  Finding the pivots of A using RREF mod p...")
    pivots = nothing
    # find pivots and check that the system is consistent (then b is not a pivot column)
    # we use deepcopy because we don't want to solve the system over ZZ 
    # (that gives bigger numbers in the pseudoinverse)
    t1 = @elapsed A2,b2 = preprocess_rows!(deepcopy(A),deepcopy(b), include_b=true)
    Ab = hcat(A2, b2)
    t = @elapsed pivots = find_pivots_modular(ZZ.(Ab))
    verbose && println(" ($(t) $(t1) s)")
    if size(Ab, 2) == pivots[end] #
        println("  The system is inconsistent; taking more columns")
        return zero_matrix(QQ, ncols(A), 1), false, false
    end
    if isnothing(pivots)
        pivots = 1:size(A,2)
    end    
    if length(pivots) < nrows(A)
        verbose && println("  We did not find enough pivots ($(length(pivots)) instead of $(nrows(A)))")
        # find linearly independent rows: 
        # (note that we can use the submatrix consisting of the pivot columns)
        rows = find_pivots_modular(transpose(ZZ.(A[:, pivots])))
    end
    
    if settings.pseudo
        try 
            # This works quite well (only 1.5 times increase in size of final solutions, vs our best)
            if settings.extracolumns_linindep
                extracolumns = []
                while sum(length.(extracolumns); init=0) + length(pivots) < settings.pseudo_columnfactor * nrows(A)
                    nonpivots = [i for i=1:size(A,2) if !(i in vcat(pivots, extracolumns...))]
                    if length(nonpivots) == 0
                        break
                    end
                    # select other columns, not overlapping with current selection
                    nonpivots = shuffle(nonpivots)
                    extra = find_pivots_modular(ZZ.(A[rows, nonpivots]))
                    push!(extracolumns, nonpivots[extra])
                end
            else
                #same format as with non-random columns
                extracolumns = [shuffle([i for i=1:size(A,2) if !(i in pivots)])] 
            end
            
            
            column_subset = unique(vcat(pivots, extracolumns...))
            #take no more than  settings.pseudo_columnfactor * nrows(A) columns, so that we can also take 1.5
            column_subset = column_subset[1:min(length(column_subset), round(Int, settings.pseudo_columnfactor * nrows(A)))]
            As = A[rows, column_subset]
            bs = b[rows, 1:1]
            x = zero_matrix(QQ, ncols(A), 1)
            
            verbose && print("  Solving the system of size $(size(As, 1)) x $(size(As, 2)) using the pseudoinverse...")
            t = @elapsed newx = solve_system_pseudoinverse(As,bs)
            x[column_subset, 1] = newx
            verbose && println(" $(t)s")
            # print_memory_info()
            correct_slacks = A * x == b
            return x, correct_slacks, true
        catch e
            println()
            println("The system is rank-deficient")
        end
    end

    #create the smaller (square) system
    Apiv = A[:, pivots]
    # print_memory_info()

    # dixon is only integers or rationals; it works but is very slow with AbsSimpleNumFieldElem (generic code)
    # It is O(n^3(log(n)+log(N))^2), where A \in R^{n x n}, and |A_ij|, |b_i| <= N
    # So it is beneficial to keep the numbers small (i.e., simple approximations everywhere)
    nA = QQ.(Apiv)
    nb = QQ.(b)
    if nrows(Apiv) != ncols(Apiv)
        verbose && println("  We solve the normal equations because the system is not square")
        nb = transpose(nA) * nb
        nA = transpose(nA) * nA
        check_slacks=true
    else
        #square system so always correct solution
        check_slacks = false
        correct_slacks = true
    end
    verbose && print("  Solve_dixon for system of size $(nrows(Apiv)) x $(ncols(Apiv))... ")
    t = @elapsed newx = Nemo._solve_dixon(nA, nb)
    verbose && println(" $(t)s")
    if check_slacks
        correct_slacks = nA * newx == nb
    end
    # print_memory_info()
    x = zero_matrix(QQ, ncols(A), 1)
    for (j,i) in enumerate(pivots)
        x[i] = newx[j]
    end

    return x, correct_slacks, true
end

function find_pivots_modular(A::ZZMatrix; maxprimes = 3)
    p = min(maximum(abs.(A)), 10^4) # start with a prime of order 10^4, or lower if that is possible
    numps = 1
    history = []
    for numps = 1:maxprimes
        p = next_prime(p)
        pivots = find_pivots_modular(A, p)
        if length(pivots) == nrows(A)
            if numps > 1
                println("  Needed to do the RREF for $numps primes")
                println("  Succeeded with p = $p")
            end
            return pivots
        elseif numps >= maxprimes
            push!(history, pivots)
            bestlength = maximum(unique(length.(history)))
            pivots = history[findfirst(x->length(x)==bestlength, history)]
            # println("The matrix does probably not have enough linearly independent columns; maximum number found is $(length(pivots))")
            return pivots
        end
        push!(history, pivots)
    end
    return nothing
end

function find_pivots_modular(A::ZZMatrix, p)
    F = FqField(ZZ(p), 1, :s)
    M = F.(A)
    rnk, res = Nemo.rref(M)
    pivots = Int[]
    for i=1:nrows(M)
        if length(pivots) > 0
            startcol = last(pivots)+1
        else
            startcol = i
        end
        for j=startcol:ncols(res)
            if !(res[i,j] == F(0))
                #first nonzero element = pivot
                push!(pivots, j)
                break
            end
        end
    end
    return unique(pivots)
end


function solve_system_pseudoinverse(A,b)
    #NOTE: this needs that A has either linearly independent columns or linearly independent rows
    # Find the minimum norm solution to Ax = b
    # If A has linearly independent rows, then A^+ = A^T (AA^T)^(-1) is the psuedoinverse
    # Then v = A^+ b is the minimum norm solution to Ax=b
    # So we need to compute v = A^T (AA^T)^(-1) b 
    # When the columns are linearly independent instead, v = (A^TA)^-1 A^T b
    At = transpose(A)
    if ncols(A) > nrows(A)
        AAt = A*At
        Dr = matrix(QQ, Diagonal([QQ(1)//gcd(Vector(numerator.(AAt[:,k]))) for k in 1:size(AAt,2)]))
        AAt = AAt * Dr
        Dl = matrix(QQ, Diagonal([QQ(1)//gcd(Vector(numerator.(AAt[k,:]))) for k in 1:size(AAt,1)]))
        AAt = Dl * AAt
        newb = Dl * b
        y = Nemo._solve_dixon(AAt, newb)
        v = At * (Dr * y)
    else
        AtA = At * A
        Dr = matrix(QQ, Diagonal([QQ(1)//gcd(Vector(numerator.(AtA[:,k]))) for k in 1:size(AtA,2)]))
        AtA = AtA * Dr
        Dl = matrix(QQ, Diagonal([QQ(1)//gcd(Vector(numerator.(AtA[k,:]))) for k in 1:size(AtA,1)]))
        AtA = Dl * AtA
        newb = Dl * (At * b)
        y = Nemo._solve_dixon(AtA, newb)
        v = Dr * y
    end
    return v
end

# g is an approximation of the real root that is used as generator
function is_valid_solution(problem::Problem, sol::PrimalSolution; check_slacks=true,  FF=QQ, g=1, verbose=true)
    success = true
    if check_slacks
        verbose && print("Checking whether slacks are zero...")
        t = @elapsed begin
            s = slacks(problem, sol)
            for i in eachindex(s)
                if !iszero(s[i])
                    success = false
                    @warn "Constraint $i is not satisfied"
                end
            end
        end
        # print_memory_info()
        if success
            verbose && print(" done")
        else
            verbose && print(" fail")
        end
        verbose && println(" ($t)")
    end

    verbose && println("Checking sdp constraints")
    # use Arb cholesky to check whether the matrices are positive definite
    t = @elapsed begin
        R, x = polynomial_ring(FF, "x")
        failures = []
        bs = blocksizes(problem)
        for k in sort(collect(keys(bs)), by=k->(bs[k], hash(k)))
            matr = sol.matrixvars[k]

            pd = checkpd_cholesky(matr; FF=FF, g=g)
            if !pd
                evs = eigvals(Matrix{BigFloat}(generic_embedding.(matr, g; base_ring=BigFloat)))
                @warn "$k failed, smallest eigenvalues are $(sort(evs)[1:min(3, length(evs))])"
                push!(failures, k)
                success = false
            end
        end
    end
    if success
        verbose && print(" done")
    else
        verbose && print(" fail")
    end
    verbose && println(" ($t)")
    # print_memory_info()
    success
end

is_negative(x::QQFieldElem, g) = x < 0

function is_negative(x::AbsSimpleNumFieldElem, gs)
    for cur_r in gs
        x_arb = sum(coeff(x,k) * cur_r^k for k=0:degree(parent(x))-1)
        if Nemo.is_negative(x_arb)
            return true
        elseif Nemo.is_nonnegative(x_arb)
            return false
        end
    end
    error("The precision is not high enough to decide whether x is positive or negative")
end

root_balls(FF::QQField, g) = [1]

function root_balls(FF, g, prec=256, max_prec=2^14)
    gs = ArbFieldElem[]
    while prec <= max_prec
        R, y = polynomial_ring(AcbField(prec), :y)
        min_poly = defining_polynomial(FF)
        rts = roots(R(min_poly), isolate_real=true, target=div(prec,2))
        rts = [real(r) for r in rts if is_real(r)] #keep the real roots
        cur_ri = argmin([abs(r-g) for r in rts]) #the real root closests to g
        push!(gs, rts[cur_ri])
        prec *=2
    end
    return gs
end

function checkpd_cholesky(A; FF=QQ, g=1, prec=256, maxprec=2^14)
    if FF != QQ
        gs = root_balls(FF, g, prec, maxprec) 
    else
        gs = [1]
    end
    i = 1
    while prec <= maxprec
        F = ArbField(prec)
        if FF != QQ
            g = F(gs[i])
            B = matrix(F, generic_embedding.(A, g, base_ring=F))
        else
            B = matrix(F, F.(A))
        end
        try 
            cholesky(B)
            return true
        catch

        end
        prec *= 2
        i += 1
    end
    return false
end


"""
    clindep(v::Matrix{T}, bits, errbound)

Find relations between the rows of v using at most `bits` bits, 
so that taking the linear combination gives a vector with entries at most errbound in absolute value.
"""
function clindep(v::Matrix{T}, bits, errbound) where T
    pow = 9
    F = AcbField(2^pow)
    u = map(F, transpose(v))
    for p = 1:5:bits
        if p >= 2^(pow-1)
            pow+=1
            F = AcbField(2^pow)
            u = map(F, transpose(v))
        end
        a = ones(Int, size(u,1))
        try 
            a = lindep(u, p)
        catch e
            # continue if the error is about lindep having not enough precision
            # that occurs if the elements of v have errorbounds and have therefore
            # not a unique integer closests to it with p bits
            # that is most probably solved by increasing p
            if !(hasfield(typeof(e), :msg)) || !occursin("precision", e.msg)
                rethrow(e)
            end            
        end
        err = maximum(abs(sum([u[k,i] * F(a[i]) for i in eachindex(a)])) for k=1:size(u,1))
        if err < errbound
            return a
        end
    end
    error("clindep failed to find a relation")
end

#round to QQ
# in the exact bounds paper code, they use Rational{BigInt}(x), which approximates the x to much higher precision
# that way, if there is a unique solution, you probably find it immediately
# this way, the right hand side of the system stays nicer if there is no unique solution
function roundx(x; power=40)
    x = [floor(BigInt, x[i]*big(10)^power)//big(10)^power for i in eachindex(x)]
end

function roundx(x, g, deg; bits=100, errbound = 1e-15)
    finalvecs = [QQFieldElem[] for d=1:deg]
    for v in x
        start_vec = [v]
        for d=0:deg-1
            push!(start_vec, g^d)
        end
        # u = map(ArbField(3000), start_vec)
        # a = lindep(u, bits)
        a = clindep(reshape(start_vec, deg+1,1), bits, errbound)
        for d=1:deg
            push!(finalvecs[d], -QQ(a[d+1] // QQ(a[1])))
        end
    end
    vcat(finalvecs...)
end

# with regularization takes much longer, but the approximate solution gets nicer
function roundx(A,b,x,g,deg; regularization=1e-30, prec=512, power=40)
    #The idea is to write x as x ~= sum_i=0^k x_i * g^i.
    # To have this almost equality, we have k variables for each entry, and let the final variable be determined by the solution for the others.
    # we choose x_0 to be dependent on the other x_i; then we can simply rationalize everything instead of dealing with 1/g^i for some i

    # use Arb balls for the matrix multiplication and solving. That is much faster than with BigFloat
    arbs = ArbField(prec)

    Af = arbs.(A)
    bf = arbs.(b)
    xf = matrix(arbs, length(x), 1, x)
    gf = arbs(g)
    n = div(nrows(b), deg)
    m = length(x)
    @assert n == div(nrows(A), deg)
    Acols = Af[:, 1:m]
    rhs = bf - Acols * xf
    # we need to distribute Ai * g^j x_j over the other columns
    for j=1:deg-1 # put Ai[j+1] g^j
        Af[:,m*j+1:m*(j+1)] -= gf ^ j * Acols
    end
    B = Af[:, m+1:end]

    lhs = transpose(B) * B + arbs(regularization) * identity_matrix(arbs, ncols(B))
    y = solve(lhs, transpose(B) * rhs, side=:right)

    #now y = x_1, x_2, ..., x_{deg-1}
    y = Matrix{BigFloat}(y) # the floating point approximation of the rational numbers
    x0 = roundx(x - sum(g^i * y[m*(i-1)+1:m*i] for i=1:deg-1), power=power)
    xfinal = vcat(x0, roundx(y, power=power))
    return xfinal
end

"""
    detecteigenvectors(dualblock, primalblock; FF=QQ, g=1, settings::RoundingSettings)

Find exact eigenvectors of `primalblock` over the field `FF` with approximate generator `g`.
"""
function detecteigenvectors(dualblock::Matrix{T}, primalblock::Matrix{T};
            FF=QQ, g=1, check_dimensions=false, settings::RoundingSettings) where T
    deg = degree(FF)
    z = gen(FF)

    # Without dualblock:
    if !settings.kernel_use_dual || maximum(abs.(dualblock)) > 1/sqrt(settings.kernel_round_errbound)
        tmp = svd(primalblock)
        num_evs = count(x->abs(x)<settings.kernel_errbound, tmp.S)
        if num_evs  == 0
            return []
        end
        mat = Matrix(transpose(tmp.U[:, end-num_evs+1:end]))
    else
        mat = copy(dualblock)
    end

    #TODO: check out whether we can do this with LU; 
    # I think the main problem is handling small numbers (considering <settings.kernel_errbound to be 0)
    RowEchelon.rref!(mat, settings.kernel_errbound)
    
    #keep only the nonzero rows; those are a basis for the kernelvectors
    vecs = [mat[i, :] for i in axes(mat,1) if norm(mat[i, :]) > settings.kernel_errbound]
    @assert all(maximum(abs.(primalblock * v)) < settings.kernel_errbound for v in vecs)
    
    #check whether the dimensions of the non-kernel vectors add up to the ambient dimension
    if check_dimensions
        mat2 = copy(primalblock)
        RowEchelon.rref!(mat2, settings.kernel_errbound)
        nonzeros = [i for i in axes(mat2, 1) if norm(mat[i,:])  > settings.kernel_errbound]
        @assert length(vecs) + length(nonzeros) == size(primalblock,1)
    end

    # round the kernel vectors entrywise
    kernel_vecs = []
    for i in eachindex(vecs)
        kv = roundx.(vecs[i], g, deg, bits=settings.kernel_bits, errbound=settings.kernel_round_errbound)
        push!(kernel_vecs, [sum(x[k+1] * z^k for k=0:deg-1) for x in kv])
    end

    # test whether kernel_vecs are really kernel vectors
    for (i,kv) in enumerate(kernel_vecs)
        kv_float = generic_embedding.(kv, g, base_ring=BigFloat)
        res = primalblock * kv_float
        if maximum(abs.(res)) > settings.kernel_errbound
            @show maximum(abs.(kv_float))
            error("Warning: wrong vector detected! (error = $(Float64.(maximum(abs.(res)))))")
            push!(removals, i)
        end
    end

    return kernel_vecs    
end

# Use LLL to find relations, and take the nullspace of that as the kernel vectors
function detecteigenvectors(block::Matrix{T}, bits::Int=10^3, errbound=1e-15; FF=QQ, g=1) where T
    tmp = svd(block)
    V = tmp.S
    M = tmp.U

    #these also work with QQField (they return 1)
    g_exact = gen(FF)
    deg = degree(FF)

    FtoQ = hcat([g^k * Matrix(I, size(M,1), size(M,1)) for k=0:deg-1]...)
    FtoQ_exact = hcat([g_exact^k .* Matrix(I, size(M,1), size(M,1)) for k=0:deg-1]...)
    M = vcat([g^k * M for k=0:deg-1]...)

    list = Vector{ZZRingElem}[]
    i = findfirst(x->abs(x)<1e-20, V[:])
    if i == nothing
        return list
    end
    m = M[:, i:size(M,2)]

    if size(block) == (1,1) && abs(block[1,1]) <= 0.000001
        list = [[1]]
    elseif size(m, 2) > 0
        A = ZZ.([0 for i=1:1, j=1:size(m,1)])
        s = collect(1:size(m,1))
        while !isempty(s)
            # l is such that transpose(m[s,:]) * l is close to zero
            l = clindep(m[s,:], bits, errbound)
            # we have an equation for every power of the generator
            if deg == 1
                new_row = ZZ.(zeros(Int, size(m,1)))
                for (idx,val) in zip(s,l)
                    new_row[idx] = val
                end
                A = [A; transpose(new_row)]
                # push!(A, sparse_row(FlintZZ, [x for x in zip(s, l) if !iszero(x[2])]))
            else
                cur_A = FtoQ_exact[:, s] * [i for i in l, j in 1:1]
                cur_b = [FF(0)]

                AQQ, bQQ = convert_system(FF, transpose(cur_A), cur_b)

                for r in eachrow(AQQ)
                    lcm_row = lcm(denominator.(r))
                    r = ZZ.(lcm_row .* r)
                    new_row = ZZ.(zeros(Int, size(m,1)))
                    for (idx,val) in enumerate(r)
                        new_row[idx] = val
                    end
                    A = [A; transpose(new_row)]
                    # push!(A, sparse_row(FlintZZ, [x for x in enumerate(r) if !iszero(x[2])]))
                end
            end
            # No idea what this is supposed to do.
            # Apparently it does something: without it it does not work properly
            # the number of columns does not change
            # push!(A, sparse_row(FlintZZ, [(i,FlintZZ(0)) for i=1:size(m,1)]))
            B = Nemo.rref(matrix(ZZ, A))[2] #rref works, but hnf doesn't?
            # B = hnf(A, truncate=true)
            nonzero_rows = length([i for i=1:nrows(B) if any(!iszero(B[i,j]) for j=1:ncols(B))])

            if size(m,1) - nonzero_rows - deg*size(m, 2) <= 0
                if size(B) == (1,1) && B[1][1] == 0
                    n = matrix(ZZ, 1, 1, [1;])
                else
                    n = nullspace(B)[2]
                end
                list = [[n[k,i] for k=1:nrows(n)] for i=1:size(n,2)]
                break
            end
            if iszero(l)
                break
            end
            deleteat!(s, findfirst(x->!iszero(x), l))
        end
    end

    removals = Int[]
    for i in eachindex(list)
        res = block * FtoQ * Vector{BigFloat}(list[i])
        # q = (size(block)[1])/2

        if maximum(abs.(res)) > 1e-8
            @show maximum(abs.(BigFloat.(list[i])))
            error("Warning: wrong vector detected! (error = $(Float64.(maximum(abs.(res)))))")
            push!(removals, i)
        end
    end
    deleteat!(list, removals)
    if length(list) != deg*size(m,2)
        display(block)
        display(list)
        @warn "Not all kernel vectors detected!"
    end
    [FtoQ_exact * v for v in list]
end

"""
    basis_transformations(dualsol, primalsol; FF=QQ, g=1, settings=RoundingSettings())

Return basis transformations for each matrix variable in primalsol such that the first few columns 
of the transformation correspond to kernel vectors. The result is a dictionary with keys corresponding to 
the variables of primalsol, and values `(transpose(B), B^-1, num_kernelvecs)` so that `(transpose(B) X B)[1:num_kernelvecs, 1:num_kernelvecs]`
is approximately zero.
"""
function basis_transformations(dualsol::DualSolution, sol::PrimalSolution; FF=QQ, g=1, settings::RoundingSettings=RoundingSettings(),verbose=true)
    Bs = Dict()
    for k in sort(collect(keys(sol.matrixvars)), by=k->(size(sol.matrixvars[k],1), hash(k)))
        m = matrixvar(sol, k)
        dm = matrixvar(dualsol, k)
        verbose && println("  Block ", k, " of size $(size(m,1)) x $(size(m,2))")
        zerolist = Int[]
        list = Vector{typeof(FF(1))}[]
        if settings.kernel_lll
            # this is only useful for the lll approach, 
            # otherwise we certainly find these
            for i = 1:size(m, 1)
                if abs(m[i, i]) < settings.kernel_errbound
                    push!(zerolist, i)
                    v = zeros(Int, size(m, 1))
                    v[i] = 1
                    push!(list, FF.(v))
                end
            end
        end
        nonzerolist = filter(x -> !(x in zerolist), collect(1:size(m, 1)))
        N = length(zerolist) + length(nonzerolist)
        m = m[nonzerolist, nonzerolist]
        dm = dm[nonzerolist, nonzerolist]
        n = size(m, 1)
        if n > 0
            # the list of eigenvectors in FF
            # these vectors might be linearly dependent over FF (but maybe not over QQ)
            # if FF = QQ, the vectors are linearly independent
            if settings.kernel_lll
                prelist = detecteigenvectors(m, settings.kernel_bits, settings.kernel_errbound; FF=FF, g=g)
                for l in prelist
                    v = FF.(zeros(Int, N))
                    v[nonzerolist] = l
                    push!(list, v)
                end
            else
                # these vectors are linearly independent over FF
                list = detecteigenvectors(dm, m; settings=settings, FF=FF, g=g)
            end
            

            # give some info about the quality by giving the minimum and maximum denominator
            # for large matrices, the kernelvectors themselves clutter the terminal too much
            if degree(FF) > 1
                coefficients = [coeff(x,k) for v in list for x in v for k=0:degree(FF)-1]
            else
                coefficients = vcat(list...)
            end
            maxnum = maximum(abs.(numerator.(coefficients)), init=0)
            maxden = maximum(denominator.(coefficients), init=0)
            verbose && length(list) > 0 && println("    Block $k has $(length(list)) kernel vectors. The maximum numerator and denominator are $maxnum and $maxden")

            if settings.kernel_lll && degree(FF) > 1
                # we only have to find linearly independent vectors
                # with the original method and a field of degree > 1.
                # With the RREF, the vectors are automatically linearly independent
                AF = ArbField(512)
                float_vecs = [generic_embedding.(v, AF(g), base_ring=AF) for v in list]
                orthogonalvectors = []
                lin_indep_list = Int[]
                for i in eachindex(float_vecs)
                    # orthogonalize with respect to the vectors in orthogonalvectors
                    for v in orthogonalvectors
                        float_vecs[i] -= (dot(v, float_vecs[i]) // dot(v,v)) .* v
                    end
                    if dot(float_vecs[i], float_vecs[i]) > 1e-40
                        push!(lin_indep_list, i)
                        push!(orthogonalvectors, float_vecs[i])
                    end
                end              
                verbose && println("  Finished GS to find linearly independent kernel vectors")
                list = list[lin_indep_list]
            end
            if length(list) > 0
                B, num_kernelvecs = simplify_kernelvectors(matrixvar(sol, k), list, FF=FF, g=g, settings=settings, verbose=verbose)
                B = FF.(B) #to have all options give a consistent type back
            else
                num_kernelvecs = 0
                B = matrix(FF, Matrix(I, N, N))
            end
        else
            prelist = []
            num_kernelvecs = length(list)
            B = matrix(FF, FF.(hcat(list...)))
        end
        Binv = Matrix(inv(B))

        #find lcm of denominators of Binv and bring that factor to B, 
        # so that the transformed problem doesn't have extra rational numbers 
        #Q: what about general fields?
        # For QQ(sqrt(2)), we have:
        # lcm(z^10, z^10) = 1024, even though z^10 = 32
        # it works, but it generates far too big numbers
        if degree(FF) == 1 && settings.normalize_transformation
            lcms = []
            for i=1:nrows(Binv)
                push!(lcms, lcm(denominator.(Binv[i,:])))
            end
            Binv = Diagonal(lcms) * Binv
            B = Matrix(B) * Diagonal([1//x for x in lcms])
        else
            B = Matrix(B)
        end

        Bs[k] = (transpose(B), Binv, num_kernelvecs)
    end
    Bs
end

function simplify_kernelvectors(dm, finalvectors; FF=QQ, g=1, settings=RoundingSettings(), verbose=true)
    # 1) go to nullspace using the structure in RREF
    # 2) Clear denominators
    # 3) Go to kernel vectors using HNF with nice transformation matrix (HNF on larger matrix)
    # 4) Use the transformation matrix as basis vectors (not only the kernel vectors but also the other vectors)
    N = length(first(finalvectors))
    FF_kerneldim = length(finalvectors)
    if degree(FF) > 1 # first go to the coefficients
        lst = [vcat([coeff.(v .* gen(FF)^i, k) for k=0:degree(FF)-1]...) for v in finalvectors for i=0:degree(FF)-1]
    else
        lst = finalvectors
    end
    initial_maximum_number = max(maximum(maximum(denominator.(v)) for v in lst), maximum(maximum(abs.(numerator.(v))) for v in lst))
    if settings.kernel_lll && settings.reduce_kernelvectors
        # we already go through the nullspace, so we only need to do the last step
        for i in eachindex(lst)
            lst[i] *= lcm(denominator.(lst[i]))
        end
        kernelvecs = transpose(matrix(ZZ, hcat(lst...)))
        # the columns of B are the kernel vectors
        B = transpose(lll(kernelvecs))
        kernel_dim = ncols(B)
        find_extra_vectors = true
    elseif settings.reduce_kernelvectors
        kernelvecs = transpose(matrix(QQ, hcat(lst...)))
        # the kernel vectors are the rows. So we need to permute the columns
        onehots = zeros(Int, nrows(kernelvecs))
        for i=1:ncols(kernelvecs) # we need 1 for every kernelvector
            succes, j = all_except1(kernelvecs,i)
            if succes && kernelvecs[j,i] == 1
                onehots[j] = i
            end
        end
        if any(==(0), onehots)
            error("The matrix of kernel vectors cannot easily be transformed to RREF")
        end
        indices = unique(vcat(onehots, 1:ncols(kernelvecs)))
        indices_rev = [findfirst(==(k), indices) for k=1:ncols(kernelvecs)]
        kernelvecs = kernelvecs[:, indices]
        if ncols(kernelvecs) > settings.reduce_kernelvectors_cutoff
            # try first to do everything by taking 2 * # kernelvecs  entries into account
            k = 1
            s = settings.reduce_kernelvectors_stepsize
            # We take the first nrows columns (identity block), plus s*k at columns after that, plus s*k columns at the end of the matrix
            # empirically, we only need 1 iteration to reduce the denominators sufficiently to get 'small' numbers
            while true
                # in each iteration, either k increases or we break the loop
                # when 2k+1 >= ncols/nrows, we certainly get an integer result, so we break.
                # so this loop certainly terminates. 
                # take (k-1)*nrows(kernelvecs) random columns
                #TODO: what do we want to do here? options:
                # - take the identity part, plus some random columns
                # - take the identity part, plus some next columns, (possibly plus some final columns)
                # cols = shuffle(nrows(kernelvecs)+1:ncols(kernelvecs))[1:(k-1)*nrows(kernelvecs)]
                # part_kernelvecs = kernelvecs[:, [1:nrows(kernelvecs)..., cols...]]
                cols = unique([1:nrows(kernelvecs)..., nrows(kernelvecs)+1:nrows(kernelvecs)+s*k..., ncols(kernelvecs)-s*k+1:ncols(kernelvecs)...])
                part_kernelvecs = kernelvecs[:, [k for k in cols if 1 <= k <= ncols(kernelvecs)]]
                kernel_dim, B_part = reduction_step(part_kernelvecs; FF=FF, g=g)
                transform = transpose(B_part[1:kernel_dim, end-kernel_dim+1:end])
                reduced_kernelvecs = transform * kernelvecs
                if all(x->denominator(x) == 1, reduced_kernelvecs)
                    # we now have an integer matrix, so we can just apply LLL
                    # note that this will happen eventually, when k gets large enough
                    reduced_kernelvecs = lll(ZZ.(reduced_kernelvecs))
                    B = transpose(reduced_kernelvecs)
                    find_extra_vectors = true
                    break
                else
                    # now we still have some noninteger vectors
                    # try get small vectors by clearing denominators 
                    # and applying LLL
                    lcms = []
                    for i=1:nrows(reduced_kernelvecs)
                        push!(lcms,lcm(denominator.(reduced_kernelvecs[i, :])))
                        reduced_kernelvecs[i, :] *= lcms[end]
                    end
                    reduced_kernelvecs = lll(ZZ.(reduced_kernelvecs))
                    maxnum = maximum(abs.(reduced_kernelvecs))
                    verbose && @info "  The transform needed for the first bunch of entries did not make the whole matrix integer. The maximum is $maxnum (compared to $initial_maximum_number)"
                    if maxnum <= initial_maximum_number
                        B = transpose(reduced_kernelvecs)
                        find_extra_vectors = true
                        break
                    else   
                        k+=1
                    end
                end
            end
        else
            kernel_dim, B = reduction_step(kernelvecs; FF=FF, g=g)
            kernelvecs = transpose(B[:, end-kernel_dim+1:end])
            kernelvecs_reduced = transpose(lll(kernelvecs))
            if settings.unimodular_transform
                # We now use the unimodular matrix B as basis transformation
                # LLL applies a unimodular transformation, so replacing the kernel vectors
                # in the basis transformation matrix B is equivalent with multiplying by a
                # unimodular matrix from the right. Hence the resulting B is still unimodular
                # but possibly with nicer basis vectors
                B[:, end-kernel_dim+1:end] = kernelvecs_reduced
                find_extra_vectors = false
            else
                B = kernelvecs_reduced
                find_extra_vectors = true
            end
        end
        # go back to the original order of the rows
        B = B[indices_rev, :]
    else
        kernel_dim = degree(FF) * FF_kerneldim
        B = matrix(QQ, hcat(lst...))
        find_extra_vectors = true
    end
    final_maxnum = maximum(abs.(B))
    if degree(FF) > 1
        # convert back to FF from QQ
        finalvectors = [ sum(B[j*N+1:(j+1)*N, i] .* gen(FF)^j for j=0:degree(FF)-1) for i=1:ncols(B)]
        # find linearly independent set of basis vectors from the given ones
        # starting at the kernel vectors, then continuing with the non kernel vectors
        AF = ArbField(512)
        float_vecs = [generic_embedding.(x, AF(g), base_ring=AF) for x in finalvectors]
        if find_extra_vectors
            # add standard basis vectors at the end
            for i=1:N
                vexact = FF.(zeros(Int, N))
                vexact[i] = FF(1)
                push!(finalvectors, vexact)
                v = AF.(zeros(Int, N))
                v[i] = AF(1)
                push!(float_vecs, v)
            end
        else
            # change the order because the kernel vectors are at the end of B
            finalvectors = [finalvectors[end-kernel_dim+1:end]..., finalvectors[1:end-kernel_dim]...]
            float_vecs = [float_vecs[end-kernel_dim+1:end]..., float_vecs[1:end-kernel_dim]...]
        end
        # check whether the first kernel_dim float_vecs are kernel vectors
        maxerror = maximum(maximum(abs.(dm * Vector{BigFloat}(v))) for v in float_vecs[1:kernel_dim])
        @assert maxerror < settings.kernel_errbound "The reduced kernel vectors are not kernel vectors (maximum error = $maxerror)"
        # orthogonalize and make new list with only the orthogonalized vectors
        indices = []
        orthogonalvectors = []
        for i in eachindex(float_vecs)
            # orthogonalize with respect to the vectors in orthogonalvectors
            for v in orthogonalvectors
                float_vecs[i] -= (dot(v, float_vecs[i]) // dot(v,v)) .* v
            end
            if dot(float_vecs[i], float_vecs[i]) > 1e-40
                push!(indices, i)
                push!(orthogonalvectors, float_vecs[i])
            end
            if length(orthogonalvectors) == N
                break
            end
        end
        finalvectors = finalvectors[indices] # kernel vectors are at the start
        B = hcat(finalvectors...)
        B = matrix(FF, B)
    elseif find_extra_vectors
        # find_extra_vectors is true, so all the current vectors should be kernel vectors
        maxerror = maximum(maximum(abs.(dm * BigFloat.(B[:, i]))) for i=1:ncols(B))
        @assert maxerror < settings.kernel_errbound "The reduced kernel vectors are not kernel vectors (maximum error = $maxerror)"
        # find remaining basis vectors
        # convert kernel vectors to floats for numerical linear independence testing
        AF = ArbField(512)
        nlist = [AF.(B[:, i]) for i=1:ncols(B)]
        extravectors = []
        # orthogonalize the kernel vectors
        for i in eachindex(nlist)
            for j=1:i-1
                nlist[i] -= (dot(nlist[i], nlist[j])// dot(nlist[j], nlist[j])) .* nlist[j]
            end
        end
        # complete the basis
        extravectors = zero_matrix(FF, N, N-ncols(B))
        j = 1 # we are looking for the j'th extra vector
        for i = 1:N
            candidate = AF.(zeros(Int, N))
            candidate[i, 1] = AF(1)
            for v in nlist
                candidate -= (dot(candidate, v) // dot(v, v)) .* v
            end
            if dot(candidate, candidate) > 1e-40
                push!(nlist, candidate)
                extravectors[i, j] = FF(1)
                j+=1
            end
            if length(nlist) == N
                break
            end
        end
        B = hcat(FF.(B), extravectors)
    else
        # we want the first bunch of vectors to be the kernel vectors;
        # by default they are at the end
        B = hcat(B[:, end-kernel_dim+1:end], B[:, 1:end-kernel_dim])
        maxerror = maximum(maximum(abs.(dm * BigFloat.(B[:, i]))) for i=1:kernel_dim)
        @assert maxerror < settings.kernel_errbound "The reduced kernel vectors are not kernel vectors (maximum error = $maxerror)"
    end
    verbose && settings.reduce_kernelvectors && println("    After reduction, the maximum number of the basis transformation matrix is $final_maxnum")
    return B, FF_kerneldim
end
function reduction_step(kernelvecs; FF=QQ, g=1)
    # go to nullspace
    ns = transpose(nullspace_fromrref(kernelvecs)[2])

    #clear denominators
    for i=1:nrows(ns)
        ns[i, :] *= lcm(Vector(denominator.(ns[i, :])))
    end
    # Go back to kernel vectors using a nice transformation
    # the last kernel_dim columns of B correspond to the kernel vectors
    kernel_dim, B = basis_nullspace_remaining(ZZ.(ns))
end

function basis_nullspace_remaining(x::ZZMatrix)
    # [H; 0] = T x^T, so x T^T = [H 0], so the last bunch of columns of T corresponds to the nullspace.
    H, T = hnf_normalmultiplier_with_transform(transpose(x))
    # H, T = hnf_with_transform(transpose(x))
    for i = nrows(H):-1:1
        for j = 1:ncols(H)
            if !iszero(H[i, j])
                #nullspace has nrows(H)-i vectors
                return nrows(H)-i, transpose(T)
            end
        end
    end
    return ncols(x), identity_matrix(x, ncols(x))
end

function hnf_normalmultiplier_with_transform(A::ZZMatrix)
    if nrows(A) < ncols(A) 
        # now the hnf and the multiplier are unique
        return hnf_with_transform(A)
    end
    # Based on Rational invariants of scalings from Hermite normal forms.
    # Evelyne Hubert, George Labahn
    # section 2.2 (we work with a different definition of HNF)
    # The idea is that appending an identity will put the nullspace part of the transformation in HNF,
    # and reduce the other part of the transformation with respect to the nullspace part
    mat = hcat(A, identity_matrix(ZZ, nrows(A)))
    H = hnf(mat)
    # V = H[:, ncols(A)+1:end]
    # H = V * A
    return H[:, 1:ncols(A)], H[:, ncols(A)+1:end]
end

function nullspace_fromrref(M::QQMatrix)
    # This is the abstractalgebra nullspace for QQ matrices, except that we check 
    # whether the matrix is in RREF and do nothing if it is.
    m = nrows(M)
    n = ncols(M)
    # rank, A = rref(M)
    if is_rref(M)
        A = M
        rank = 0
        for i=1:m
            for j=1:n
                # rank increases if there is a nonzero element in this row
                if A[i,j] != 0
                    rank+=1
                    break
                end
            end
        end
    else
        rank, A = Nemo.rref(M)
    end
    nullity = n - rank
    R = base_ring(M)
    X = zero(M, n, nullity)
    if rank == 0
        for i = 1:nullity
            X[i, i] = one(R)
        end
    elseif nullity != 0
        pivots = zeros(Int, max(m, n))
        np = rank
        j = k = 1
        for i = 1:rank
            while is_zero_entry(A, i, j)
                pivots[np + k] = j
                j += 1
                k += 1
            end
            pivots[i] = j
            j += 1
        end
        while k <= nullity
            pivots[np + k] = j
            j += 1
            k += 1
        end
        for i = 1:nullity
            for j = 1:rank
                X[pivots[j], i] = -A[j, pivots[np + i]]
            end
            X[pivots[np + i], i] = one(R)
        end
    end
    return nullity, X
end

function all_except1(m,k)
    # return true, i if there is exactly 1 nonzero element in column k, on position i
    tot = 0
    indx = 0
    for i=1:nrows(m)
        if m[i,k] != 0 
            if tot == 1
                return false, indx
            end
            indx = i
            tot += 1
        end
    end
    if tot == 0
        return false, indx
    end
    return true, indx
end

# TODO: first submatrix of Binv, then multiply
function transform(p::LowRankMatPol, Binv, s)
    vs = [(Binv*v)[s+1:end] for v in p.vs]
    ws = [(Binv*v)[s+1:end] for v in p.ws]
    LowRankMatPol(p.lambda, vs, ws)
end

#TODO: check if this is needed
# Nemo.copy(a::QQFieldElem) = deepcopy(a) 

function transform(p::Matrix{T}, Binv, s;g=1) where T
    if T == BigFloat
        Binv = generic_embedding.(Binv,g, base_ring=BigFloat)
    end
    Binv[s+1:end, :]*p*transpose(Binv)[:, s+1:end]
end

function transform(p::MatrixElem, Binv, s;g=1)
    transform(Matrix(p), Binv, s, g=g)
end

"""
    transform(problem, Bs)

Transform the problem using the basis transformations Bs obtained from `basis_transformations`.
"""
function transform(problem::Problem, Bs)
    matrixcoeff = Dict()
    for (k, m) in problem.objective.matrixcoeff
        if Bs[k][3] < size(m, 1)
            matrixcoeff[k] = transform(m, Bs[k][2], Bs[k][3])
        end
    end
    objective = Objective(problem.objective.constant, matrixcoeff, problem.objective.freecoeff)

    constraints = []
    for constraint in problem.constraints
        matrixcoeff = Dict()
        for (k, m) in constraint.matrixcoeff
            if Bs[k][3] < size(m, 1)
                matrixcoeff[k] = transform(m, Bs[k][2], Bs[k][3])
            end
        end
        push!(constraints, Constraint(constraint.constant, matrixcoeff, constraint.freecoeff, constraint.samples, constraint.scalings))
    end
    Problem(problem.maximize, objective, constraints)
end

function transform(sol::PrimalSolution{T}, Bs; g=1) where T
    matrixcoeff = Dict{Any,Matrix{T}}()
    for (k, m) in sol.matrixvars
        if Bs[k][3] < size(m, 1)
            matrixcoeff[k] = transform(m, Bs[k][1], Bs[k][3], g=g)
        end
    end
    PrimalSolution{T}(sol.base_ring, matrixcoeff, sol.freevars)
end

function undo_transform(sol::PrimalSolution{T}, Bs; FF=QQ) where T
    matrixcoeff = Dict{Any,Matrix{T}}()
    for k in keys(Bs)
        N = size(Bs[k][1], 1)
        M = [sol.base_ring(0) for i=1:N, j=1:N]
        if haskey(sol.matrixvars, k)
            M[Bs[k][3]+1:end, Bs[k][3]+1:end] = sol.matrixvars[k]
            matrixcoeff[k] = transform(M, transpose(Bs[k][2]), 0)
            # matrixcoeff[k] = transform(M, Matrix(inv(matrix(FF, Bs[k][1]))), 0)
        else
            matrixcoeff[k] = M
        end
    end
    PrimalSolution{T}(sol.base_ring, matrixcoeff, sol.freevars)
end


function convert_system(FF, A, b)
    # convert A from FF to QQ, where FF is some number field with a generator
    # 1) split A, x, b into sum_i A_i g^i, where g is the generator
    # 2) combine A_i,x_j to get b_k -> find blocks of new system
    # 3) make system by multiplying by inverse of g^k
    g = gen(FF)
    Ai = [matrix(QQ, coeff.(A, k)) for k=0:degree(FF)-1]
    btot = vcat([coeff.(b, k) for k=0:degree(FF)-1]...)
    # btot = matrix(QQ, length(btot), 1, btot)

    n, m = size(A)
    Atot = zero_matrix(QQ, n*degree(FF), m*degree(FF))
    for i=0:degree(FF)-1
        for j=0:degree(FF)-1
            # A_i * x_j * g^(i+j) = \sum_{k} A_i * x_j * alpha_{k,i,j} g^k
            # where alpha_{k,i,j} depends on the minimal polynomial
            cur_gen = g^(i+j)
            for k=0:degree(FF)-1
                if coeff(cur_gen, k) != 0
                    #add A_i to the k-th row, j-th column of the block-matrix
                    Atot[n*k+1:n*(k+1), m*j+1:m*(j+1)] += coeff(cur_gen,k) * Ai[i+1]
                end
            end
        end
    end
    return Atot, btot
end

function get_rational_system(problem::Problem, primalsol::PrimalSolution, FF::QQField, g=1; columns = nothing, monomial_bases=nothing, settings=RoundingSettings(), verbose=true)
    # It would be best if linearsystem would return type information
    if isnothing(columns) # take all columns
        columns = 1:length(vectorize(primalsol))
    end
    x = vectorize(primalsol)
    x = roundx(x, power=settings.approximation_decimals)
    verbose && print("  Constructing the linear system...")
    t = @elapsed A, b = partial_linearsystem(problem, as_primal_solution(primalsol, x), columns; monomial_bases=monomial_bases)
    verbose && println(" ($(t)s)")

    return A, b, x, columns
end

#Here g is a numerical approximation of the generator of FF
function get_rational_system(problem::Problem, primalsol::PrimalSolution, FF::AbsSimpleNumField, g; columns=nothing, monomial_bases=nothing, settings=RoundingSettings(), verbose=true)
    #make sure that g is indeed an approximation of the generator
    p = defining_polynomial(FF)
    val = sum(BigFloat(coeff(p,k)) * g^k for k=0:degree(p))
    @assert abs(val) < 1e-10
    # get the linear system and convert it
    # Note that now we need to get the whole system, since we need to get an approximate solution to the extended system
    verbose && print("  Constructing the linear system...")
    t = @elapsed if isnothing(monomial_bases)
        A, b = linearsystem(problem, FF=FF)
    else
        A, b = linearsystem_coefficientmatching(problem, monomial_bases, FF=FF)
    end
    verbose && println(" ($(t)s)")
    A = FF.(A)
    b = FF.(b)
    nvars = ncols(A)
    A, b = convert_system(FF, A, b)
    x = vectorize(primalsol)

    verbose && print("  Computing an approximate solution in the extension field...")
    t = @elapsed x_rounded = roundx(A, b, x, g, degree(FF), regularization=settings.regularization, prec=Int64(precision(first(x))), power=settings.approximation_decimals)
    verbose && println(" ($(t)s)")
    # convert to error vector and smaller system
    b -= A*matrix(QQ, length(x_rounded), 1, x_rounded)
    if !isnothing(columns)
        columns = [i+nvars*k for i in columns for k=0:degree(FF)-1]
        A = A[:, columns]
    end
    # we want to return QQMatrix, QQMatrix, Vector{Rational{(Big)Int}}
    return A, b, x_rounded, columns
end

function xrational_to_field(x, FF)
    # x is the concatenation of x_j, where sum_j=0 x_j * g^j is the value in FF
    g = gen(FF)
    d = degree(FF)
    x_FF = FF.(zeros(Int, div(length(x),d)))
    for k=0:d-1
        x_FF += g^k .* x[length(x_FF)*k+1:length(x_FF)*(k+1)]
    end
    return x_FF
end

function print_memory_info()
    mem = Sys.total_memory() - Sys.free_memory()
    if mem > 2^30
        factor = 2^30
        s = "Gb"
    else
        factor = 2^20
        s = "Mb"
    end
    println("Total memory usage: $(mem/factor) $s")
end

#TODO: extra argument: known kernel vectors. This would
# 1) allow for the use of lower precision solutions if all kernel vectors are known
# 2) reduce the time needed for the step of finding all kernel vectors, since presumably a bunch of the computations can be skipped
# For problems in extremal geometry, many of the kernel vectors come from one or multiple configurations, and this can then be easily automated.
# We probably need this to round the exact solution for the 12-point codes in S^3, or we need more precision (60-80 bits instead of 40)
"""
    exact_solution(problem::Problem, dualsol::DualSolution, primalsol::PrimalSolution; transformed=false, FF = QQ, g=1, settings::RoundingSettings=RoundingSettings(), monomial_bases=nothing)

Compute and return an exact solution to the problem, given a dual solution, primal solution and a field `FF` with approximate generator `g`.
Return `(success, exactprimalsol)` if `transformed=false`, and `(success, pd_transformed_exactsolution, transformations)` if `transformed=true`.
"""
function exact_solution(problem::Problem, dualsol::DualSolution, primalsol::PrimalSolution; 
            transformed=false, 
            FF = QQ, g=1, 
            settings::RoundingSettings=RoundingSettings(),
            monomial_bases=nothing,
            verbose=true)
    verbose && println("** Starting computation of basis transformations **")
    # print_memory_info()
    t = @elapsed Bs = basis_transformations(dualsol, primalsol, FF=FF, g=g, settings=settings, verbose=verbose)
    verbose && println("** Finished computation of basis transformations ($(t)s) **")
    # print_memory_info()
    verbose && print("** Transforming the problem and the solution **")
    problem = map(x->generic_embedding(x, gen(FF); base_ring=FF), problem)
    if !isnothing(monomial_bases)
        monomial_bases = [map(x->generic_embedding(x, gen(FF); base_ring=FF), m) for m in monomial_bases]
        @assert length(monomial_bases) == length(constraints(problem)) "Please supply one monomial basis for every constraint."
        length.(monomial_bases) != [length(x.samples) for x in constraints(problem)] && @warn "Some constraints have a different number of basis polynomials than samples."
    end
    # verbose && print("Transforming primalsol... ")
    t = @elapsed transformed_primalsol = transform(primalsol,Bs, g=g)
    # verbose && println("done ($(t)s)")
    # print_memory_info()
    # verbose && print("Transforming problem... ")
    t += @elapsed transformed_problem = transform(problem, Bs)
    verbose && println(" ($(t)s)")
    # print_memory_info()
	verbose && println("** Projection the solution into the affine space **")
    t = @elapsed exact_sol, correct_slacks = project_to_affine_space(transformed_problem, transformed_primalsol, FF=FF, g=g, settings=settings, monomial_bases=monomial_bases, verbose=verbose)
	verbose && println("** Finished projection into affine space ($(t)s) **")
    verbose && println("** Checking feasibility **")
    if !correct_slacks
        verbose && println("The slacks are nonzero")
    else
        verbose && println("The slacks are satisfied (checked or ensured by solving the system)")
    end
    success = is_valid_solution(transformed_problem, exact_sol, FF=FF, g=g, check_slacks=false, verbose=verbose)
    success = success && correct_slacks
    final_transform = Dict(k=>transpose(Binv)[:, s+1:end] for (k, (Bt,Binv,s)) in Bs)
    if transformed
        return success, exact_sol, final_transform
    else
        return success, undo_transform(exact_sol, Bs; FF=FF)
    end
end

# #TODO: replace to_arb and to_BigFloat by generic_embedding?
# # This basically is generic_embedding.
# # Q: do we want generic_embedding in the solver interface?
# # That makes it much easier to work with number fields
# function to_arb(x::AbsSimpleNumFieldElem, g)
#     sum(coeff(x,k) * g^k for k=0:degree(parent(x))-1)
# end
# function to_arb(x::QQFieldElem, g::ArbFieldElem)
#     parent(g)(x)
# end
# # For converting an exact problem to a floating point SDP
# #TODO: see to_arb (replace by generic_embedding?)
# function to_BigFloat(x::AbsSimpleNumFieldElem,g)
#     F = parent(x)
#     sum(BigFloat(coeff(x,k))*g^k for k=0:degree(F)-1)
# end
# function to_BigFloat(x::QQFieldElem,g)
#     BigFloat(x)
# end

# go to a new polynomial ring
# we need this for transforming the problem if the kernel vectors have non-rational elements
function Base.:*(x::AbsSimpleNumFieldElem, y::QQMPolyRingElem)
    FF = parent(x)
    R = parent(y)
    n = length(gens(R))
    R2, z = polynomial_ring(FF, n)
    P = MPolyBuildCtx(R2)
    # y2 = R2(0)
    for (c,ev) in zip(coefficients(y), exponent_vectors(y))
        push_term!(P, x * FF(c), ev)
        # y2 += x * FF(c) * prod(z .^ ev)
    end
    return finish(P)
end


