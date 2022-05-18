#########################
### Low rank matrices ###
#########################

struct LowRankMat
    eigenvalues::Vector{Arb}
    leftevs::Vector{ArbRefMatrix}
    rightevs::Vector{ArbRefMatrix}
end
function LowRankMat(eigenvalues, eigenvectors)
    LowRankMat(eigenvalues, eigenvectors, eigenvectors)
end

Base.getindex(A::LowRankMat,i,j) = sum(A.eigenvalues[r] * A.leftevs[r][j] * A.rightevs[r][i] for r=1:length(A.eigenvalues))

function Base.size(m::LowRankMat, i)
    length(m.leftevs[1])
end

function convert_to_prec(A::LowRankMat,prec=precision(BigFloat))
    evs = [Arb(ev,prec=prec) for ev in A.eigenvalues]
    leftvecs = [ArbRefMatrix(vec,prec=prec) for vec in A.leftevs]
    rightvecs = [ArbRefMatrix(vec,prec=prec) for vec in A.rightevs]
    for i=1:length(evs)
        Arblib.get_mid!(evs[i],evs[i])
        Arblib.get_mid!(leftvecs[i],leftvecs[i])
        Arblib.get_mid!(rightvecs[i],rightvecs[i])
    end
    return LowRankMat(evs,leftvecs,rightvecs)
end

######################################
### Completely sampled polynomials ###
######################################


struct SampledMPolyRing{T} <: AbstractAlgebra.Ring
    base_ring
end

SampledMPolyRing(ring::T) where T<:AbstractAlgebra.Ring = SampledMPolyRing{elem_type(ring)}(ring)

==(a::SampledMPolyRing{T}, b::SampledMPolyRing{T}) where T = true # a.base_ring == b.base_ring TODO: this is a workaround

struct SampledMPolyElem <: AbstractAlgebra.RingElem
    evaluations::Dict
end

function SampledMPolyElem(p::MPolyElem, samples)
    SampledMPolyElem(Dict(samples[s] => p(samples[s]...) for s in eachindex(samples)))
end

function SampledMPolyElem(samples, values)
    SampledMPolyElem(Dict(samples[s] => values[s] for s in eachindex(samples)))
end

function evaluate(p::SampledMPolyElem, v...)
    p.evaluations[collect(v)]
end

function (p::SampledMPolyElem)(v...)
    evaluate(p, v...)
end

# addition
function Base.:+(p::SampledMPolyElem, q::SampledMPolyElem)
    @assert keys(p.evaluations) == keys(q.evaluations)
    SampledMPolyElem(Dict(sample => p.evaluations[sample] + q.evaluations[sample] for sample in keys(p.evaluations)))
end

function Base.:+(p::MPolyElem, q::SampledMPolyElem)
    r = copy(q)
    for s in keys(r.evaluations)
        r.evaluations[s] += p(s...)
    end
    r
end

function Base.:+(q::SampledMPolyElem, p::MPolyElem)
    p + q
end

# subtraction
function Base.:-(q::SampledMPolyElem)
    SampledMPolyElem(Dict(sample => -q.evaluations[sample] for sample in keys(q.evaluations)))
end

function Base.:-(p::T, q::SampledMPolyElem) where T <: Union{SampledMPolyElem, MPolyElem}
    p + (-q)
end

function Base.:-(p::SampledMPolyElem, q::MPolyElem)
    p + (-q)
end

# multiplication
function Base.:*(p::MPolyElem, q::SampledMPolyElem)
    r = copy(q)
    for s in keys(r.evaluations)
        r.evaluations[s] *= p(s...)
    end
    r
end

function Base.:*(p, q::SampledMPolyElem)
    r = copy(q)
    for s in keys(r.evaluations)
        r.evaluations[s] *= p(s...)
    end
    r
end

function Base.:*(q::SampledMPolyElem, p::MPolyElem)
    p * q
end

function Base.:*(p::SampledMPolyElem, q::SampledMPolyElem)
    @assert keys(p.evaluations) == keys(q.evaluations)
    r = copy(q)
    for s in keys(r.evaluations)
        r.evaluations[s] *= p(s...)
    end
    r
end

function Base.copy(q::SampledMPolyElem)
    SampledMPolyElem(copy(q.evaluations))
end

"""
    approximatefekete(basis, samples) -> basis, samples

Compute approximate fekete points based on samples and a corresponding orthogonal basis.
The basis consists of sampled polynomials, sampled at `samples`

This preserves a degree ordering of `basis` if present.
"""
function approximatefekete(basis, samples;show_det=false,s=3)
    V, P, samples = approximate_fekete(samples, basis,show_det=show_det,s=s)
    [SampledMPolyElem(samples, V[:,p]) for p in eachindex(basis)], samples
end


###################################
### Low rank matrix polynomials ###
###################################

polys = Union{MPolyElem, SampledMPolyElem}

"""
    LowRankMat(eigenvalues::Vector, rightevs::Vector{Vector}[, leftevs::Vector{Vector}])

The matrix ``∑_i λ_i v_i w_i^T``.

If `leftevs` is not specified, use `leftevs = rightevs`.
"""
struct LowRankMatPol
    eigenvalues::Vector{polys}
    leftevs::Vector{Vector{polys}}
    rightevs::Vector{Vector{polys}}
end

function LowRankMatPol(eigenvalues, eigenvectors)
    LowRankMatPol(eigenvalues, eigenvectors, eigenvectors)
end
LinearAlgebra.transpose(A::Union{LowRankMatPol,LowRankMat}) = typeof(A)(A.eigenvalues,A.rightevs,A.leftevs)
Base.getindex(A::LowRankMatPol,i,j) = sum(A.eigenvalues[r] * A.leftevs[r][j] * A.rightevs[r][i] for r=1:length(A.eigenvalues))

function Base.size(m::LowRankMatPol, i)
    length(m.leftevs[1])
end

function AbstractAlgebra.evaluate(p::LowRankMatPol, sample...; scaling = 1, prec=precision(BigFloat))
    LowRankMat([Arb(scaling*v(sample...), prec=prec) for v in p.eigenvalues],
               [ArbRefMatrix([v(sample...) for v in w], prec=prec) for w in p.leftevs],
               [ArbRefMatrix([v(sample...) for v in w], prec=prec) for w in p.rightevs])
end


#############################
### Low rank SOS problems ###
#############################

"""
    Block(l::Any[, r::Int, s::Int])

Specifies a block corresponding to the positive semidefinite variable `l`.

Specifying `r,s` makes the Block correspond to the `r,s` subblock of the variable `l`.
"""
struct Block
    l::Any
    r::Int
    s::Int
end

function Block(l)
    Block(l, 1, 1)
end

"""
    Constraint(constant, matrixcoeff, freecoeff, samples[, scalings])

Models a polynomial constaint of the form
```math
    c(x) = ∑_l ⟨ Y^l_{r,s}, A^l_{r,s}(x) ⟩ + ∑_i y_i B_i(x)
```
with samples, where `r,s` are defined by the `Block` structure with `l` as first argument

Arguments:
  - `constant::MPolyElem`
  - `matrixcoeff::Dict{Block, LowRankMatPol}`
  - `freecoeff::Dict{Any, MPolyElem}`
  - `samples::Vector{Vector}`
  - `scalings::Vector`
"""
struct Constraint
    constant::MPolyElem
    matrixcoeff::Dict{Block,LowRankMatPol}
    freecoeff::Dict{Any,MPolyElem}
    samples::Vector{Vector}
    scalings::Vector
end
Constraint(constant,matrixcoeff,freecoeff,samples) = Constraint(constant,matrixcoeff,freecoeff,samples,[1 for s in samples])


"""
    Objective(constant, matrixcoeff::Dict{Block, Matrix}, freecoeff::Dict)

The objective for the LowRankPolProblem
"""
struct Objective
    constant
    matrixcoeff::Dict{Block,Matrix}
    freecoeff::Dict
end

"""
    LowRankPolProblem(maximize, objective, constraints)

Combine the objective and constraints into a low-rank polynomial problem
"""
struct LowRankPolProblem #Maybe we  need to rename this to e.g. LowRankPolProblem
    maximize::Bool
    objective::Objective
    constraints::Vector{Constraint}
end

##############################
### Clustered low rank SDP ###
##############################

struct ClusteredLowRankSDP
    maximize::Bool
    constant::Arb #constant is part of the objective, so maybe we should have constant, C, b, A, B, c or something like that.
    A::Vector{Vector{Matrix{Dict{Int,LowRankMat}}}}         # A[j][l][r,s][p]
    B::Vector{ArbRefMatrix}                                 # B[j]
    c::Vector{ArbRefMatrix}                                 # c[j]
    C::BlockDiagonal{ArbRef,BlockDiagonal{ArbRef,ArbRefMatrix}} # C[j][l]
    b::ArbRefMatrix                                         #b
    matrix_coeff_names::Vector{Vector{Any}}                 # the (user defined) names of the blocks
    free_coeff_names::Vector{Any}                           # the (user defined) names of the free variables
end

function ClusteredLowRankSDP(maximize,constant,A,B,c,C,b)
    matrixcoeff_names = [[(j,l) for l in eachindex(A[j])] for j in eachindex(A)]
    freecoeff_names = collect(1:size(b,1))
    return ClusteredLowRankSDP(maximize,constant,A,B,c,C,b,matrixcoeff_names,freecoeff_names)
end

"""
    ClusteredLowRankSDP(sos::LowRankPolProblem; as_free::Vector, prec, verbose])

Define a ClusteredLowRankSDP based on the LowRankPolProblem sos.

The PSD variables defined by the keys in the vector `as_free` will be modelled as extra free variables,
with extra constraints to ensure that they are equal to the entries of the PSD variables.
Remaining keyword arguments:
  - `prec` (default: precision(BigFloat)) - the precision of the result
  - `verbose` (default: false) -  print progress to the standard output
"""
function ClusteredLowRankSDP(sos::LowRankPolProblem; as_free = [], prec=precision(BigFloat),verbose = false)
    # the blocks in as_free will be modelled as extra variables
    if length(as_free) > 0
        if verbose
            println("Modelling the selected PSD variables as free variables...")
        end
        #We do not modify the sos problem
        sos = deepcopy(sos)
        for l in as_free
            #For each constraint, for each Block b with l==b.l, we need to
            #   1) add the entries of the LowRankMatPol to the freecoeff dict with key b,i,j
            #   2) remove the block from the constraint (from the dict matrixcoeff)
            #For each block we use, we need to:
            #   1) add a new constraint equating the free variable with the entry of the block
            m = 0
            n = 0
            for constraint in sos.constraints
                for block in collect(keys(constraint.matrixcoeff))
                    if block.l == l #we do this l now
                        if n == 0
                            n = size(constraint.matrixcoeff[block],1)
                        elseif n != size(constraint.matrixcoeff[block],1)
                            error("The blocks are not of equal sizes")
                        end
                        for i=1:size(constraint.matrixcoeff[block],1)
                            for j=1:size(constraint.matrixcoeff[block],2)
                                if block.r == block.s && i >= j
                                    #diagonal (r,s) block, so we only need to do things when i >= j
                                    if i == j # just the entry
                                        constraint.freecoeff[(l,(block.r-1)*n+i,(block.s-1)*n+j)] = constraint.matrixcoeff[block][i,j]
                                    else #i > j, so we need to take both sides of the diagonal into account
                                        constraint.freecoeff[(l,(block.r-1)*n+i,(block.s-1)*n+j)] = 2*constraint.matrixcoeff[block][i,j]
                                    end
                                elseif block.r > block.s #the other case is r < s, but because it is symmetric we do that now too. (hence the factor 2)
                                    constraint.freecoeff[(l,(block.r-1)*n+i,(block.s-1)*n+j)] = 2 * constraint.matrixcoeff[block][i,j]
                                end
                            end
                        end
                        m = max(block.r,block.s,m)
                        delete!(constraint.matrixcoeff,block)
                    end
                end
            end


            # add equality constraints for the new free variables and the new blocks
            R, (x,) = PolynomialRing(RealField,["x"])
            for i=1:n*m
                for j=1:i
                    if i==j
                        push!(sos.constraints,Constraint(R(0),
                            Dict(Block(l,i,j)=>LowRankMatPol([R(1)],[[R(1)]])), # matrix variable
                            Dict((l,i,j)=>R(-1)), # free variable
                            [[0]],
                            ))
                    else # i != j, so both sides get 1/2
                        push!(sos.constraints,Constraint(R(0),
                            Dict(Block(l,i,j) => LowRankMatPol([R(1)],[[R(1)]]), Block(l,j,i) => LowRankMatPol([R(1)],[[R(1)]])),
                            Dict((l,i,j)=>R(-2)),
                            [[0]],
                        ))
                    end
                end
            end

            # change the objective
            # Two options:
            # 1) hvcat the blocks and set an objective for the new subblocks (PSD side)
            # 2) set an objective for the free variables.
            # 3) set both to 1/2 * objective
            #This is an implementation of (1)
            new_blocks = Dict() #because we delete the blocks afterwards, so we need to store the new blocks separately and put them in the objective afterwards
            for block in collect(keys(sos.objective.matrixcoeff)) # now this is not modified during the loop
                if block.l == l && block.r >= block.s # we only do the lower half, and we make it symmetric
                    for i=1:n
                        for j=1:(block.r == block.s ? i : n)
                            if (block.r-1)*n+i == (block.s-1)*n + j # on the diagonal
                                new_blocks[Block(l, (block.r-1)*n+i, (block.s-1)*n+j)] = hcat(sos.objective.matrixcoeff[block][i,j])
                            else #not on the diagonal of the full matrix, so we need to take both sides into account
                                new_blocks[Block(l, (block.r-1)*n+i, (block.s-1)*n+j)] = hcat(sos.objective.matrixcoeff[block][i,j])
                                new_blocks[Block(l, (block.s-1)*n+j, (block.r-1)*n+i)] = hcat(sos.objective.matrixcoeff[block][i,j])
                            end
                        end
                    end
                end
                if block.l == l
                    delete!(sos.objective.matrixcoeff,block)
                end
            end
            for (block, value) in new_blocks
                sos.objective.matrixcoeff[block] = value
            end

        end
    end
    if verbose
        println("Forming the clusters...")
    end
    # clusters will be a vector of vectors containing block labels (so block.l)
    #NOTE: if a constraint doesn't have PSD variables, this creates an empty cluster.
    clusters = Vector{Any}[]
    for constraintindex in eachindex(sos.constraints)
        constraint = sos.constraints[constraintindex]
        clusterset = Set{Int}()
        for k in keys(constraint.matrixcoeff)
            for i in eachindex(clusters)
                if k.l in clusters[i]
                    push!(clusterset, i)
                end
            end
        end
        clusterset = sort(collect(clusterset))
        if isempty(clusterset)
            if length(keys(constraint.matrixcoeff)) > 0 #only create a new cluster for constraints with matrix blocks
                push!(clusters, [k.l for k in keys(constraint.matrixcoeff)])
            end
        else
            #remove the clusters which we need to combine from 'clusters', and merge them
            u = union(splice!(clusters, clusterset)...)
            append!(u, [k.l for k in keys(constraint.matrixcoeff)])
            push!(clusters, u)
        end
    end

    #NOTE: if a constraint doesn't use PSD variables, this will not assign it to a cluster.
    # find out which constraints belong to which cluster
    # we build a vector of vectors containing constraint indices
    cluster_constraint_index = [Int[] for _ in clusters]
    remaining = Int[]
    for constraintindex in eachindex(sos.constraints)
        constraint = sos.constraints[constraintindex]
        k = first(keys(constraint.matrixcoeff))
        found_cluster = false
        for clusterindex in eachindex(clusters)
            if k.l in clusters[clusterindex]
                push!(cluster_constraint_index[clusterindex], constraintindex)
                found_cluster = true
                break
            end
        end
        if !found_cluster
            #this is a constraint without cluster
            push!(remaining,constraintindex)
        end
    end
    if length(remaining)>0
        # there is at least on constraint which doesn't use PSD variables, we put it in a separate cluster (otherwise we would forget about it)
        #NOTE: this is untested, since we did not encounter this in any problem
        push!(cluster_constraint_index, remaining)
    end
    if verbose
        println("Sampling the low rank polynomial matrices...")
    end
    A = []
    C = []
    matrix_coeff_names = []
    for clusterindex in eachindex(cluster_constraint_index)
        constraintindices = cluster_constraint_index[clusterindex]
        nsubblocks = Dict{Any,Int}()
        subblocksizes = Dict{Any,Int}()
        for constraintindex in constraintindices
            constraint = sos.constraints[constraintindex]
            for block in keys(constraint.matrixcoeff)
                subblocksizes[block.l] = max(size(constraint.matrixcoeff[block], 1),get(subblocksizes,block.l,0))
                nsubblocks[block.l] = max(block.r, block.s, get(nsubblocks, block.l, 0))
            end
        end
        v = [[Dict{Int,LowRankMat}() for _=1:m, _=1:m] for (k, m) in nsubblocks]
		subblock_keys = collect(keys(nsubblocks))

        i = 1
        for constraintindex in constraintindices
            constraint = sos.constraints[constraintindex]
            for (idx,sample) in enumerate(constraint.samples) #This can take quite long for a large amount of samples & a large basis
                for block in keys(constraint.matrixcoeff)
                    v[findfirst(isequal(block.l), subblock_keys)][block.r,block.s][i] = evaluate(constraint.matrixcoeff[block], sample...,scaling = constraint.scalings[idx],prec=prec)
                    # We can set the s,r block here to the transpose of the r,s block. Or to 1/2 * the transpose if we would want that.
                end
                i += 1
            end
        end

        push!(A, v)

        M = [[ArbRefMatrix(subblocksizes[l], subblocksizes[l],prec=prec) for r=1:m, s=1:m] for (l, m) in nsubblocks]
        for block in keys(sos.objective.matrixcoeff)
            if block.l in clusters[clusterindex]
                M[findfirst(isequal(block.l), subblock_keys)][block.r, block.s] = ArbRefMatrix(sos.objective.matrixcoeff[block],prec=prec)
            end
        end
        push!(C, [ArbRefMatrix(hvcat(size(S,2), [S[r,s] for r=1:size(S,1) for s=1:size(S,2)]...),prec=prec) for S in M])
        push!(matrix_coeff_names, subblock_keys)
    end
    C_block = BlockDiagonal([BlockDiagonal([b for b in blocks]) for blocks in C])

    if verbose
        println("Sampling the coefficients of the free variables...")
    end

    freecoefflabels = Set()
    for constraint in sos.constraints
        union!(freecoefflabels, keys(constraint.freecoeff))
    end
    freecoefflabels = collect(freecoefflabels) #We want an ordering for B and b

    # B is a vector of the matrices B^j
    B = ArbRefMatrix[]
    for constraintindices in cluster_constraint_index
        Bj = []
        for constraintindex in constraintindices
            constraint_B = ArbRefMatrix(length(sos.constraints[constraintindex].samples),length(freecoefflabels),prec=prec)
            for s_idx in eachindex(sos.constraints[constraintindex].samples)
                s = sos.constraints[constraintindex].samples[s_idx]
                scaling = sos.constraints[constraintindex].scalings[s_idx]
                for (k,v) in sos.constraints[constraintindex].freecoeff
                    constraint_B[s_idx,findfirst(isequal(k),freecoefflabels)] = Arb(scaling*v(s...))
                end
            end
            push!(Bj,constraint_B)
        end
        push!(B,vcat(Bj...))
    end
    if verbose
        println("Sampling the constants...")
    end
    c = [ArbRefMatrix([sos.constraints[constraintindex].scalings[s_idx]*sos.constraints[constraintindex].constant(sample...) for constraintindex in constraintindices for (s_idx,sample) in enumerate(sos.constraints[constraintindex].samples)],prec=prec) for constraintindices in cluster_constraint_index]

    b = ArbRefMatrix(length(freecoefflabels),1,prec=prec)
    for (k,v) in sos.objective.freecoeff
        b[findfirst(isequal(k), freecoefflabels)] = Arb(v)
    end
    ClusteredLowRankSDP(sos.maximize, Arb(sos.objective.constant,prec=prec), A, B, c, C_block, b, matrix_coeff_names,freecoefflabels)
end

function convert_to_prec(sdp::ClusteredLowRankSDP,prec=precision(BigFloat))
    # We convert everything to the new precision
    new_A = similar(sdp.A)
    new_B = similar(sdp.B)
    new_C = similar(sdp.C)
    new_c = similar(sdp.c)
    new_constant = Arb(sdp.constant,prec=prec)
    for j in eachindex(sdp.A)
        new_A[j] = similar(sdp.A[j])
        for l in eachindex(sdp.A[j])
            new_A[j][l] = similar(sdp.A[j][l])
            for r=1:size(sdp.A[j][l],1)
                for s=1:size(sdp.A[j][l],2)
                    new_A[j][l][r,s] = Dict{Int,LowRankMat}()

                    for p in keys(sdp.A[j][l][r,s])
                        new_A[j][l][r,s][p] = convert_to_prec(sdp.A[j][l][r,s][p], prec)
                    end
                end
            end
            new_C.blocks[j].blocks[l] = ArbRefMatrix(sdp.C.blocks[j].blocks[l],prec=prec)
            Arblib.get_mid!(new_C.blocks[j].blocks[l], new_C.blocks[j].blocks[l])
        end
        new_B[j] = ArbRefMatrix(sdp.B[j], prec=prec)
        Arblib.get_mid!(new_B[j], new_B[j])

        new_c[j] = ArbRefMatrix(sdp.c[j], prec=prec)
        Arblib.get_mid!(new_c[j], new_c[j])
    end

    new_b = ArbRefMatrix(sdp.b, prec=prec)
    Arblib.get_mid!(new_b, new_b)

    return ClusteredLowRankSDP(sdp.maximize,new_constant,new_A,new_B,new_c,new_C,new_b,sdp.matrix_coeff_names,sdp.free_coeff_names)
end


######################
### Solver results ###
######################

# Define the possible status
abstract type Status end
struct Optimal <: Status end
struct NearOptimal <: Status end
struct PrimalFeasible <: Status end
struct DualFeasible <: Status end
struct Feasible <: Status end
struct NotConverged <: Status end

optimal(::Status) = false
optimal(::Optimal) = true


#The name to make clear what results these are. Distinguishing it from e.g. SDPA-GMP if someone uses both?
#TODO: decide whether to change CLRSResults to ClusteredLowRankSolverResults
#TODO: should we remove the y and the Y now that we gave the names back
#TODO: should we convert x, X to BigFloat too? or keep matrixvar etc in Arbs?
struct CLRSResults
    #the internally used matrices
    x::ArbRefMatrix
    X::BlockDiagonal{ArbRef,BlockDiagonal{ArbRef,ArbRefMatrix}}
    y::ArbRefMatrix
    Y::BlockDiagonal{ArbRef,BlockDiagonal{ArbRef,ArbRefMatrix}}
    #the objectives
    primal_objective::BigFloat
    dual_objective::BigFloat
    #the dual solutions with the user-defined names
    matrixvar::Dict{Any, Matrix{BigFloat}}
    freevar::Dict{Any, BigFloat}
    # matrix_coeff_names::Vector{Vector{Any}}
    # free_coeff_names::Vector{Any}
    # status::Status #optimal, near optimal, primal feasible, dual feasible, feasible (p&d),not converged
end
function CLRSResults(x,X,y,Y, primal_objective,dual_objective, matrix_coeff_names::Vector, free_coeff_names::Vector)
    #convert to
    matrixvar = Dict(m => BigFloat.(mv) for j=1:length(matrix_coeff_names) for (m,mv) in zip(matrix_coeff_names[j], Y.blocks[j].blocks) )
    freevar = Dict(f => BigFloat(fv) for (f,fv) in zip(free_coeff_names,y))
    return CLRSResults(x,X,y,Y, BigFloat(primal_objective),BigFloat(dual_objective), matrixvar, freevar)
end

Base.show(io::IO, x::Optimal) = @printf(io, "pdOpt")
Base.show(io::IO, x::NearOptimal) = @printf(io,"NearOpt")
Base.show(io::IO, x::Feasible) = @printf(io,"pdFeas")
Base.show(io::IO, x::PrimalFeasible) = @printf(io,"pFeas")
Base.show(io::IO, x::DualFeasible) = @printf(io,"dFeas")
Base.show(io::IO, x::Status) = @printf(io,"NOINFO")
