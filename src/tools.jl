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
    if n==1 || size(A,1)*size(A,2)*size(B,2) * div(prec,256) < 6^3
        #TODO: determine up to what sizes (& what number of threads) running multithreaded is better than threaded.
        return Arblib.approx_mul!(C,A,B,prec=prec)
    end
    # determine how to do threading
    if !(opt in [1,2,3])
        #we don't really want to do the inner (option 3), but if that size is more than twice as large, we do that.
        opt = argmax([size(B,2),size(A,1),1//2*size(A,2)]) 
    end

    # opt == 1 => distribute the columns of B over threads (preferred above rows when equal due to julia/arb being column-major)
    # opt == 2 => distribute the rows of A over threads
    # opt == 3 => distribute the columns of A/rows of B over threads and add the results
    # Option 1 and 2 are preferred above 3, because in those cases we don't have to add matrices. argmax gives the first maximal element
    # for now, we'll just do one of the options. Maybe in some cases it is better to have for example a 3x3 grid instead of 1x9, so to use multiple options instead of 1
    if opt == 1
        thread_func_right_window!(C,A,B,prec=prec)
    elseif opt == 2
        thread_func_left_window!(C,A,B,prec=prec)
    elseif opt == 3
        thread_func_inner_window!(C,A,B,prec=prec)
    end
    return C
end

function thread_func_inner_window!(C::T,A::T,B::T;n=Threads.nthreads(),prec=precision(C)) where T <: Union{ArbMatrix, ArbRefMatrix}
    # distribute B over several cores and apply func!(C[part],A,B[part],prec=prec)
    nc = size(B,1)
    sizes = [div(nc,n) for i=1:n] .+ [div(nc,n)*n+i <= nc ? 1 : 0 for i=1:n]
    idxs = [0, cumsum(sizes)...]
    # idx_cols = [idxs[i]+1:idxs[i+1] for i=1:n]

    parts_A = [T(0,0) for i=1:n]
    parts_B = [T(0,0) for i=1:n]
    part_res = [similar(C) for i=1:n] # when n_inner =1, this is max one copy of C
    Threads.@threads for i=1:n
        Arblib.window_init!(parts_A[i], A, 0,idxs[i], size(A,1), idxs[i+1])
        Arblib.window_init!(parts_B[i], B, idxs[i],0, idxs[i+1], size(B,2))
        Arblib.approx_mul!(part_res[i],parts_A[i],parts_B[i],prec=prec)
        Arblib.window_clear!(parts_A[i])
        Arblib.window_clear!(parts_B[i])
    end
    for i=1:n
        Arblib.add!(C, C, part_res[i])
    end
end
function thread_func_left_window!(C::T,A::T,B::T;n=Threads.nthreads(),prec=precision(C)) where T <: Union{ArbMatrix, ArbRefMatrix}
    # distribute B over several cores and apply func!(C[part],A,B[part],prec=prec)
    nr = size(A,1)
    sizes = [div(nr,n) for i=1:n] .+ [div(nr,n)*n+i <= nr ? 1 : 0 for i=1:n]
    idxs = [0, cumsum(sizes)...]
    # idx_rows = [idxs[i]+1:idxs[i+1] for i=1:n]

    parts_A = [T(0,0) for i=1:n]
    part_res = [T(0,0) for i=1:n] # when n_inner =1, this is max one copy of C
    Threads.@threads for i=1:n
        Arblib.window_init!(parts_A[i], A, idxs[i], 0, idxs[i+1], size(A,2))
        Arblib.window_init!(part_res[i], C, idxs[i], 0, idxs[i+1], size(C,2))
        Arblib.approx_mul!(part_res[i],parts_A[i],B,prec=prec)
        Arblib.window_clear!(parts_A[i])
        Arblib.window_clear!(part_res[i])
    end
end

function thread_func_right_window!(C::T,A::T,B::T; n =Threads.nthreads(), prec=precision(C)) where T <: Union{ArbMatrix, ArbRefMatrix}
    # distribute B over several cores and apply func!(C[part],A,B[part],prec=prec)
    nc = size(B,2)
    sizes = [div(nc,n) for i=1:n] .+ [div(nc,n)*n+i <= nc ? 1 : 0 for i=1:n]
    idxs = [0, cumsum(sizes)...]

    parts_B = [T(0,0, prec=prec) for i=1:n]
    part_res = [T(0,0,prec=prec) for i=1:n] # when n_inner =1, this is max one copy of C
    Threads.@threads for i=1:n
        Arblib.window_init!(parts_B[i], B, 0, idxs[i], size(B,1), idxs[i+1])
        Arblib.window_init!(part_res[i], C, 0, idxs[i], size(C,1), idxs[i+1])
        Arblib.approx_mul!(part_res[i],A,parts_B[i],prec=prec)
        Arblib.window_clear!(parts_B[i])
        Arblib.window_clear!(part_res[i])
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



