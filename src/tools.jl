# This file includes several general functions which helps the solver.
# This includes:
# - threaded matrix multiplication (matmul_threaded)
# - symmetrizing an ArbRefMatrix (symmetric!)
# - inner product for ArbRefMatrices & BlockDiagonals (LinearAlgebra.dot)
# - Arblib Cholesky without error bounds (approx_cholesky!)

# extending LinearAlgebra.dot for the BlockDiagonal matrices and Arb(Ref)Matrices.
# In principle dot(A,B) is also (mathematically) defined if they have different block sizes, and can be done more efficient than materalizing the matrices.
# But in that case it is far more difficult to implement, so we just give the normal matrix results then
function LinearAlgebra.dot(A::BlockDiagonal, B::BlockDiagonal)
    # tot = Arb(prec=precision(A))
    # for (a,b) in zip(blocks(A),blocks(B))
    #     Arblib.add!(tot,tot,dot(a,b))
    # end
    # return tot
    if length(blocks(A)) == length(blocks(B))
        return sum(dot(a, b) for (a, b) in zip(blocks(A), blocks(B)))
    else
        return dot(Matrix(A),Matrix(B))
    end
end

#Extending the LinearAlgebra.dot for ArbMatrices. Arblib does not have an inner product for matrices
function LinearAlgebra.dot(A::T, B::T) where T <: Union{ArbMatrix,ArbRefMatrix}
    #This is also faster for ArbMatrices due to the addmul instead of += A[i,j]*B[i,j], which allocates
    @assert size(A) == size(B)
    res = Arb(0, prec = precision(A))
    for i = 1:size(A, 1)
        for j = 1:size(A, 2) #Arb is row major?
            Arblib.addmul!(res, A[i,j], B[i,j])
        end
    end
    return res
end

function LinearAlgebra.transpose(A::T;prec=precision(A)) where T<:Union{ArbRefMatrix,ArbMatrix}
    B = T(size(A,2),size(A,1),prec=prec)
    Arblib.transpose!(B,A)
    return B
end

function symmetric!(A::ArbRefMatrix,uplo=:U)
    #make the matrix symmetric, using the upper triangular part (uplo=:U) or the lower triangular part otherwise
    size(A,1) == size(A,2) || error("A must be square")

    for j=1:size(A,2)
        for i=j+1:size(A,1)
            if uplo == :U
                A[i,j] = A[j,i]
            else
                A[j,i] = A[i,j]
            end
        end
    end
    A
end

function approx_submul!(a::T,b::T,c::T, prec=precision(a)) where T <: Union{Arb,ArbRef}
    Arblib.submul!(Arblib.midref(a), Arblib.midref(b), Arblib.midref(c), prec=prec)
end

function approx_div!(a::T,b::T,c::T, prec=precision(a)) where T <: Union{Arb,ArbRef}
    Arblib.div!(Arblib.midref(a), Arblib.midref(b), Arblib.midref(c),prec= prec)
end
function approx_sqrt!(a::T,b::T, prec=precision(a)) where T <: Union{Arb,ArbRef}
    Arblib.sqrt!(Arblib.midref(a), Arblib.midref(b),prec= prec)
end
function approx_cholesky!(A::ArbRefMatrix,B::ArbRefMatrix;prec=precision(A))
    size(A) == size(B) || error("A and B must have the same size")
    Arblib.set!(A,B)
    approx_cholesky!(A,prec=prec)
end

function approx_cholesky!(A::ArbRefMatrix;prec=precision(A))
    #The cholesky of Arblib, but with zeroed error bounds during the calculations
    size(A,1) == size(A,2) || error("The matrix must be square")
    n = size(A,1)
    Arblib.get_mid!(A,A)
    for i=1:n
        for j=1:i-1
            for k=1:j-1
                approx_submul!(A[i,j], A[i,k], A[j,k], prec)
            end
            approx_div!(A[i,j], A[i,j], A[j,j], prec)
            # Arblib.get_mid!(A[i,j],A[i,j])
        end
        for k=1:i-1
            approx_submul!(A[i,i], A[i,k], A[i,k], prec)
        end
        # Arblib.get_mid!(A[i,i], A[i,i])
        if Arblib.is_positive(A[i,i]) == 0
            @show A[i,i], i
            return 0
        end
        approx_sqrt!(A[i,i], A[i,i], prec)
        # Arblib.get_mid!(A[i,i], A[i,i])
    end

    #zero the (strict) upper triangular part:
    for i=1:n
        for j=i+1:n
            A[i,j] = 0
        end
    end
    return 1
end


function unique(x::Vector{T}) where T<:Union{ArbMatrix,ArbRefMatrix}
    unique_idx = Int[]
    for i=1:length(x)
        #We compare this one with the elements we did already
        duplicate = false
        for j=1:i-1
            if x[i] == x[j]
                # this is a duplicate
                duplicate = true
                break
            end
        end
        if !duplicate
            push!(unique_idx,i)
        end
    end
    return x[unique_idx]
end

function matmul_threaded!(C::T, A::T, B::T; n = Threads.nthreads(), prec=precision(C), opt::Int=0) where T<:Union{ArbMatrix, ArbRefMatrix}
    # matrix multiplication C = A*B, using multithreading.
    # Note that this is not the same as Arblib.mul_threaded!(), because that uses the classical matrix multiplication with flint threads
    # We will have to tune the parameters, because Arblib does classical mat mul when some dimension of the matrix has size <= 40 (i.e. A has <= 40 rows or columns, or B has <=40 rows or columns)
    # We might have very rectangular matrices (i.e., the number of rows much larger than the number of columns or vice versa), in which case it is easy do distribute over threads. When the number of remaining rows/columns can be below their cutoff due to our threading, we might want to consider using less threads? or just call the block version directly?

    # n = Threads.nthreads()
    #check dimensions
    @assert size(C,1) == size(A,1) && size(C,2) == size(B,2) && size(A,2) == size(B,1)
    #check for the special case of 1 thread:
    if n==1 || size(A,1)*size(A,2)*size(B,2) * div(prec,256) < 1000
        #TODO: determine up to what sizes (& what number of threads) running multithreaded is better than threaded.
        return Arblib.approx_mul!(C,A,B,prec=prec)
    end
    # determine how to do threading
    if !(opt in [1,2,3,4])
        opt = argmax([size(B,2),size(A,1),size(A,2)])
    end

    # opt == 1 => distribute the columns of B over threads (preferred above rows when equal due to julia/arb being column-major)
    # opt == 2 => distribute the rows of A over threads
    # opt == 3 => distribute the columns of A/rows of B over threads and add the results
    # Option 1 and 2 are preferred above 3, because in those cases we don't have to add matrices. argmax gives the first maximal element
    # for now, we'll just do one of the options. Maybe in some cases it is better to have for example a 3x3 grid instead of 1x9, so to use multiple options instead of 1
    # That will just be generating different idx_rows, idx_cols and inner stuff

    if opt == 1
        thread_func_right!(C,A,B,Arblib.approx_mul!,prec=prec)
        return C
    else
        idx_cols = [1:size(B,2) for i=1:n]
    end
    if opt == 2
        thread_func_left!(C,A,B,Arblib.approx_mul!,prec=prec)
        return C
    else
        idx_rows = [1:size(A,1) for i=1:n]
    end

    #If we are threading over the outer part, we only need one result. So n_inner gives the number of results, idx_inner gives the indices thread i uses, res_idx gives which result is used
    if opt == 3
        n_inner = n
        nr = size(A,2)
        # every thread gets at least floor(nr/n) rows, and we add the remaining parts until they are all distributed
        sizes = [div(nr,n_inner) for i=1:n_inner] .+ [div(nr,n_inner)*n_inner+i <= nr ? 1 : 0 for i=1:n_inner]
        idxs = [0, cumsum(sizes)...]
        idx_inner = [idxs[i]+1:idxs[i+1] for i=1:n_inner]
        res_idx = [i for i=1:n_inner]
    else
        n_inner = 1
        idx_inner = [1:size(A,2) for i=1:n]
        res_idx = [1 for i=1:n]
    end

    if opt == 4 #both over rows and columns, no inner. This option can only be chosen manually. It seems to be far less efficient
        # we want the blocks to be about square. So we need to find numbers a, b such that rows/a ≈ cols/b and a*b ≈ n
        if size(C,1) == size(C,2) #square, so we take squareroot
            sqrtn = isqrt(n) #integer squareroot, so sqrtn^2 < n probably. We
            if sqrtn^2 != n
                a = sqrtn
                b = sqrtn+1
            else
                a = b = sqrtn
            end
        else
            r = size(C,1)
            c = size(C,2)
            a = isqrt(div(n*r,c))
            b = div(n,a)
            #it is possible that we can increase a by one while keeping a*b <= n
            a = max(a, div(n,b))
        end
        n = a*b #new n based on the partition

        rsizes =  [div(size(C,1),a) for i=1:a] .+ [div(size(C,1),a)*a+i <= size(C,1) ? 1 : 0 for i=1:a]
        ridxs = [0, cumsum(rsizes)...]
        idx_rows_part = [ridxs[i]+1:ridxs[i+1] for i=1:a]

        csizes =  [div(size(C,2),b) for i=1:b] .+ [div(size(C,2),b)*b+i <= size(C,2) ? 1 : 0 for i=1:b]
        cidxs = [0, cumsum(csizes)...]
        idx_cols_part = [cidxs[i]+1:cidxs[i+1] for i=1:b]

        #final partition: iteration i does block C[idx_rows[i],idx_cols[i]]
        idx_rows = [idx_rows_part[div(i,b)+1] for i=0:n-1]
        idx_cols = [idx_cols_part[i%b+1] for i=0:n-1]

        n_inner = 1
        idx_inner = [1:size(A,2) for i=1:n]
        res_idx = [1 for i=1:n]
    end

    # Do the threaded matrix multiplication
    # We could use 1 matrix less if we use C with n_inner > 1 too.
    if n_inner > 1
        res_matrices = [T(size(C,1),size(C,2),prec=prec) for i=1:n_inner]
    end
    part_res = [T(length(idx_rows[i]),length(idx_cols[i]),prec=prec) for i=1:n] # when n_inner =1, this is max one copy of C
    As = [A[idx_rows[i],idx_inner[i]] for i=1:n] #in total max n copies of A, but in total 1 copy if n_inner = 1
    Bs = [B[idx_inner[i],idx_cols[i]] for i=1:n] # in total max n copies of B
    Threads.@threads for i=1:n
        # cur_res = T(length(idx_rows[i]),length(idx_cols[i]),prec=prec)
        # I am not sure whether allocating inside a threaded loop + finalizers can give memory leakage.
        # for ArbMatrices we would probably want to use the ref versions of the slice & set operations
        # Arblib.approx_mul!(part_res[i],A[idx_rows[i],idx_inner[i]],B[idx_inner[i],idx_cols[i]]) #do we want to use a different mul here?
        Arblib.approx_mul!(part_res[i],As[i],Bs[i])
        if n_inner > 1 # race conditions if we set/add to C directly, so we set the parts
            res_matrices[res_idx[i]][idx_rows[i],idx_cols[i]] = part_res[i]
        else
            C[idx_rows[i],idx_cols[i]] = part_res[i] # we actually would want to use a view/window of C[...,...] for the matrix product. But the Arblib window does not work nicely due to init/clear and normal view does not work with ArbMatrices.
        end
    end

    #add the results if needed
    if n_inner > 1
        # this way we don't have to zero C, we do that implicitely.
        # if n_inner>1, then we at least have 2 matrices to add
        Arblib.add!(C,res_matrices[1],res_matrices[2],prec=prec)
        for i=3:n_inner
            Arblib.add!(C,C,res_matrices[i],prec=prec)
        end
    end
    return C
end
function thread_func_right!(C::T,A::T,B::T,func!;n=Threads.nthreads(),prec=precision(C)) where T <: AbstractMatrix
    # distribute B over several cores and apply func!(C[part],A,B[part],prec=prec)
    nc = size(B,2)
    sizes = [div(nc,n) for i=1:n] .+ [div(nc,n)*n+i <= nc ? 1 : 0 for i=1:n]
    idxs = [0, cumsum(sizes)...]
    idx_cols = [idxs[i]+1:idxs[i+1] for i=1:n]

    parts_B = [B[:,idx_cols[i]] for i=1:n]
    part_res = [T(size(C,1),length(idx_cols[i]),prec=prec) for i=1:n] # when n_inner =1, this is max one copy of C
    Threads.@threads for i=1:n
        func!(part_res[i],A,parts_B[i],prec=prec)
        C[:,idx_cols[i]] = part_res[i]
    end
end

function thread_func_left!(C::T,A::T,B::T,func!;n=Threads.nthreads(),prec=precision(C)) where T <: AbstractMatrix
    # distribute B over several cores and apply func!(C[part],A,B[part],prec=prec)
    nr = size(A,1)
    sizes = [div(nr,n) for i=1:n] .+ [div(nr,n)*n+i <= nr ? 1 : 0 for i=1:n]
    idxs = [0, cumsum(sizes)...]
    idx_rows = [idxs[i]+1:idxs[i+1] for i=1:n]

    parts_A = [A[idx_rows[i],:] for i=1:n]
    part_res = [T(length(idx_rows[i]),size(C,2),prec=prec) for i=1:n] # when n_inner =1, this is max one copy of C
    Threads.@threads for i=1:n
        func!(part_res[i],parts_A[i],B,prec=prec)
        C[idx_rows[i],:] = part_res[i]
    end
end

function malloc_trim(k::Int)
    if Sys.islinux()
        #max because the parameter has to be at least 0
        ccall(:malloc_trim,Cvoid,(Cint,),max(k,0))
    end
end

# For BlockDiagonals it is not clear what the precision is, because different blocks have potentially a different precision
# So we make sure not to extend the Base.precision function for BlockDiagonals.
# However, then we need to define precision for ArbMatrices and BigFloats
precision(P::BlockDiagonal) = precision(P.blocks[1]) #we assume that all blocks have the same precision
precision(x) = Base.precision(x) #for other things, just use the Base version



