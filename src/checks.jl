# checks on the SDP and the polynomial problem
# 1) check whether the matrices in the SDP are symmetric
# 2) check whether the low-rank matrices have a valid decomposition \sum_i l_i v_i w_i^T
# 3) check whether every constraint has a PSD matrix
# 4) check whether all variables in the objective are used in the constraints


"""
Check whether all matrices used in the semidefinite program are symmetric
"""
function LinearAlgebra.issymmetric(sdp::ClusteredLowRankSDP;eps=1e-10)
    issym = true
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            #objective:
            if !issymmetric(sdp.C.blocks[j].blocks[l])
                @warn "The coefficient for the block $(sdp.matrix_coeff_names[j][l]) is not symmetric in the objective"
            end
            #constraints:
            for r=1:size(sdp.A[j][l],1)
                for s=1:r
                    for p in keys(sdp.A[j][l][r,s])
                        #matrix itself should be symmetric on the diagonal
                        if r == s & !issymmetric(sdp.A[j][l][r,s][p],eps=eps)
                            @warn "The coefficient for the block $(sdp.matrix_coeff_names[j][l]), subblock ($r, $s), is not symmetric in one of the constraints"
                            issym = false
                        elseif r != s && (!(p in keys(sdp.A[j][l][s,r])) || !istranspose(sdp.A[j][l][r,s][p], sdp.A[j][l][s,r][p],eps=eps))
                            @warn "The coefficient for the block $(sdp.matrix_coeff_names[j][l]), subblock ($r, $s), is not equal to the transpose of the subblock ($s,$r) in one of the constraints"
                            issym = false
                        end
                    end
                end
            end
        end
    end

    return issym
end
function LinearAlgebra.issymmetric(A::Union{ArbRefMatrix, LowRankMat}; eps=1e-10)
    n = size(A,1)
    @assert size(A,2) == n
    for i=1:n
        for j=1:i-1
            if abs(A[i,j] - A[j,i] ) > eps
                return false
            end
        end
    end
    return true
end

function istranspose(A,B; eps=1e-10)
    @assert size(A) == size(B)[end:-1:1]
    for i=1:size(A,1)
        for j=1:size(A,2)
            if abs(A[i,j] - B[j,i]) > eps
                return false
            end
        end
    end
    return true
end

function remove_empty_mats!(sdp::ClusteredLowRankSDP, verbose=true)
    toberemoved = [[] for j in eachindex(sdp.A)]
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            isempty = false
            for r=1:size(sdp.A[j][l],1)
                for s=1:r
                    for p in keys(sdp.A[j][l][r,s])
                        if is_empty(sdp.A[j][l][r,s][p])
                            isempty = true
                            delete!(sdp.A[j][l][r,s], p)
                        end
                    end
                end
            end
            verbose && isempty && @info "The coefficient for the PSD variable $(sdp.matrix_coeff_names[j][l]) has an empty decomposition in a constraint, so we remove it from that constraint."
            if all(length(sdp.A[j][l][r,s]) == 0 for r=1:size(sdp.A[j][l],1) for s=1:r)
                @info "The matrix variable $(sdp.matrix_coeff_names[j][l]) is not used in any constraint and will be removed."
                push!(toberemoved[j], l)
            end
        end
    end
    for j in eachindex(sdp.A)
        length(toberemoved[j]) == 0 && continue
        deleteat!(sdp.matrix_coeff_names[j], toberemoved[j])
        deleteat!(sdp.matrix_coeff_blocks[j], toberemoved[j])
        deleteat!(sdp.A[j], toberemoved[j])
        deleteat!(sdp.C.blocks[j].blocks, toberemoved[j]) 
    end
end
"""
Check if the matrix A is empty or the zero matrix
"""
function is_empty(A::LowRankMat)
    return length(A.ws) == 0 || length(A.ws[1]) == 0 || all(iszero, A.lambda) || all(x->all(iszero, x), A.ws) || all(x->all(iszero, x), A.vs)
end
function is_empty(A::ArbRefMatrix)
    return size(A) == (0,0) || iszero(A)
end


function check_mat(A::Union{LowRankMat, LowRankMatPol})
    #same number of vectors and values, same lengths of vectors
    correct = length(A.ws) == length(A.vs) == length(A.lambda) && all(length(v) == length(A.ws[1]) == length(w)  for (v,w) in zip(A.ws, A.vs))
    if !correct
        @info "A coefficient matrix does not have a correct decomposition (it needs the same number of vectors as values, and the vectors need to be of the same length)"
    end
    return correct
end
check_mat(A::AbstractMatrix) = size(A,1) == size(A,2)

"""
    check_sdp!(sdp::ClusteredLowRankSDP)

Check whether the constraint matrices are symmetric, and remove empty constraint matrices.
"""
function check_sdp!(sdp::ClusteredLowRankSDP, verbose=true)
    #perform all checks on the SDP
    everythingokay = issymmetric(sdp)
    remove_empty_mats!(sdp, verbose)
    return everythingokay
end

"""
    check_problem(problem::LowRankPolProblem)

Check for obvious mistakes in the constraints and objective
"""
function check_problem(prob::Problem)
    everythingokay = true
    #perform checks on the problem
    for c in prob.constraints
        everythingokay =  everythingokay && check_constraint(c)
    end
    everythingokay && check_objective(prob)
end

"""
Check whether the objective uses variables that are also used in the constriants
"""
function check_objective(prob::Problem)
    all_found = true
    for p in keys(prob.objective.matrixcoeff)
        key_found = false
        for c in prob.constraints
            if name(p) in [name(k) for k in keys(c.matrixcoeff)]
                key_found = true
                break
            end
        end
        if !key_found
            @warn "The positive semidefinite variable $(name(p)) is used in the objective but not in the constraints."
            all_found = false
        end
    end
    for p in keys(prob.objective.freecoeff)
        key_found = false
        for c in prob.constraints
            if p in keys(c.freecoeff)
                key_found = true
                break
            end
        end
        if !key_found
            @warn "The free variable $(p) is used in the objective but not in the constraints."
            all_found = false
        end
    end
    return all_found
end
"""
Check whether the constraint includes PSD variables
"""
function check_constraint(constraint::Constraint)
    everythingokay = true
    for (k,v) in constraint.matrixcoeff
        everythingokay = everythingokay && check_mat(v)
    end
    if length(constraint.matrixcoeff) == 0
        @warn "This constraint does not use any positive semidefinite variables"
        everythingokay = false
    end
    return everythingokay
end
    