#########################
### Missing functions ###
#########################

Base.precision(::Nemo.AbstractAlgebra.Floats{BigFloat}) = precision(BigFloat)

evaluate(a::Rational{Int}, x) = a

# Whether this function is needed seems to depend on the version of Nemo
# Base.Rational{BigInt}(a::QQFieldElem) = BigInt(numerator(a)) // BigInt(denominator(a))

######################################
### Completely sampled polynomials ###
######################################

struct SampledMPolyRing{T} <: Nemo.Ring
    base_ring
    samples
    function SampledMPolyRing(base_ring, samples)
        @assert issorted(samples)
        T = typeof(base_ring(1))
        new{T}(base_ring, samples)
    end
end

==(a::SampledMPolyRing, b::SampledMPolyRing) = a.base_ring == b.base_ring && a.samples == b.samples

struct SampledMPolyRingElem{T} <: Nemo.RingElem
    parent::SampledMPolyRing{T}
    evaluations::AbstractVector{T}
end

==(p::SampledMPolyRingElem, q::SampledMPolyRingElem) = parent(p) == parent(q) && p.evaluations == q.evaluations

function Base.deepcopy_internal(q::SampledMPolyRingElem, d::IdDict)
    if haskey(d, q)
        return d[q]
    end
    newq = SampledMPolyRingElem(parent(q), Base.deepcopy_internal(q.evaluations,d))
    d[q] = newq
    return newq
end

Nemo.elem_type(::Type{SampledMPolyRing{T}}) where T = SampledMPolyRingElem{T}
Nemo.parent_type(::Type{SampledMPolyRingElem{T}}) where T = SampledMPolyRing{T}

Base.parent(p::SampledMPolyRingElem) = p.parent

Nemo.base_ring(R::SampledMPolyRing) = R.base_ring

(R::SampledMPolyRing{T})(x) where T = SampledMPolyRingElem{T}(R,[R.base_ring(x) for _ in R.samples])
(R::SampledMPolyRing{T})() where T = R(0)

function (R::SampledMPolyRing)(p::SampledMPolyRingElem)
    if parent(p) == R
        return p
    end
    SampledMPolyRingElem(R, [R.base_ring(p(x...)) for x in R.samples])
end

Base.one(R::SampledMPolyRing{T}) where T = R(1)
Base.zero(R::SampledMPolyRing{T}) where T = R(0)

function Base.zero(p::SampledMPolyRingElem{T}) where T
    zero(parent(p))
end
function Base.one(p::SampledMPolyRingElem{T}) where T
    one(parent(p))
end

(R::SampledMPolyRing)(p::MPolyRingElem) = SampledMPolyRingElem(R, [R.base_ring(p(x...)) for x in R.samples])

Base.iszero(p::SampledMPolyRingElem{T}) where T = all(iszero, p.evaluations)
Base.isone(p::SampledMPolyRingElem{T}) where T = all(isone, p.evaluations)

Nemo.canonical_unit(p::SampledMPolyRingElem) = one(p)

function show(io::IO, R::SampledMPolyRing)
    print(io, "Sampled polynomials in $(length(first(R.samples))) variables over ")
    show(io, base_ring(R))
 end
 
 function show(io::IO, p::SampledMPolyRingElem)
    print(io, "Sampled polynomial in ")
    show(io, parent(p))
    print(io, " with evaluations ")
    show(io, p.evaluations)
 end

function (p::PolyRingElem{T})(sp::SampledMPolyRingElem{T}) where T <: RingElem
    r = zero(sp)
    for si in eachindex(parent(sp).samples)
        r.evaluations[si] = p(sp.evaluations[si])
    end
    return r        
end
function (p::QQPolyRingElem)(sp::SampledMPolyRingElem{T}) where T <: RingElem
    r = zero(sp)
    for si in eachindex(parent(sp).samples)
        r.evaluations[si] = p(sp.evaluations[si])
    end
    return r        
end

function (p::Generic.MPoly{T})(sp::Vararg{SampledMPolyRingElem{T}}) where T <: RingElement
    length(sp) == nvars(parent(p)) || ArgumentError("Number of variables does not match number of values")
    @assert all(q.parent == sp[1].parent for q in sp)
    r = zero(sp[1])
    for si in eachindex(r.evaluations)
        r.evaluations[si] = p([q.evaluations[si] for q in sp]...)
    end
    return r
end

function evaluate(p::SampledMPolyRingElem, v)
    i = searchsortedfirst(parent(p).samples, v)
    @assert parent(p).samples[i] == v || parent(p).samples[i] === v
    p.evaluations[i]
end

function (p::SampledMPolyRingElem)(v...)
    evaluate(p, collect(v))
end

# addition
function Base.:+(p::SampledMPolyRingElem{T}, q::SampledMPolyRingElem{T}) where T
    parent(p) != parent(q) && error("Incompatible rings")
    SampledMPolyRingElem(parent(p),p.evaluations .+ q.evaluations)
end

function Base.:+(p::MPolyRingElem, q::SampledMPolyRingElem)
    parent(q)(p) + q
end

function Base.:+(q::SampledMPolyRingElem, p::MPolyRingElem)
    p + q
end

# subtraction
function Base.:-(q::SampledMPolyRingElem)
    SampledMPolyRingElem(parent(q),-q.evaluations)
end

function Base.:-(p::T, q::SampledMPolyRingElem) where T <: Union{SampledMPolyRingElem, MPolyRingElem}
    p + (-q)
end

function Base.:-(p::SampledMPolyRingElem, q::MPolyRingElem)
    p + (-q)
end

# multiplication
function Base.:*(p::Union{MPolyRingElem{T}, AbstractFloat, Rational{Int}, Int, RingElem}, q::SampledMPolyRingElem{T}) where T
    parent(q)(p) * q
end

function Base.:*(q::SampledMPolyRingElem{T},p::Union{MPolyRingElem{T}, AbstractFloat, Rational{Int}, Int, RingElem}) where T
    p * q
end

function Base.:*(p::SampledMPolyRingElem{T}, q::SampledMPolyRingElem{T}) where T
    parent(p) != parent(q) && error("Incompatible rings")
    SampledMPolyRingElem(parent(p), p.evaluations .* q.evaluations)
end

function Base.:^(p::SampledMPolyRingElem, n::Int)
    if n == 0
        return one(p)
    elseif n < 0
        throw(DomainError(n, "Negative exponents are not allowed for SampledMPolyRingElem's"))
    else
        return SampledMPolyRingElem(parent(p), [x^n for x in p.evaluations])
    end
end

#unsafe operators
function Nemo.zero!(q::SampledMPolyRingElem)
    for i in eachindex(q.evaluations)
        q.evaluations[i] = zero(base_ring(parent(q)))
    end
    return q
end

function Nemo.mul!(p::SampledMPolyRingElem{T}, q::SampledMPolyRingElem{T}, r::SampledMPolyRingElem{T}) where T
    parent(p) == parent(q) == parent(r) || error("Incompatible rings")
    for i in eachindex(p.evaluations)
        p.evaluations[i] = q.evaluations[i] * r.evaluations[i]
    end
    return p
end

function Nemo.add!(p::SampledMPolyRingElem{T}, q::SampledMPolyRingElem{T}, r::SampledMPolyRingElem{T}) where T
    parent(p) == parent(q) == parent(r) || error("Incompatible rings")
    for i in eachindex(p.evaluations)
        p.evaluations[i] = q.evaluations[i] + r.evaluations[i]
    end
    return p
 end

 function Nemo.addeq!(p::SampledMPolyRingElem{T}, q::SampledMPolyRingElem{T}) where T
    parent(p) == parent(q) || error("Incompatible rings")
    for i in eachindex(p.evaluations)
        p.evaluations[i] += q.evaluations[i]
    end
    return p
 end

"""
    approximatefekete(basis, samples) -> basis, samples

Compute approximate fekete points based on samples and a corresponding orthogonal basis.
The basis consists of sampled polynomials, sampled at `samples`

This preserves a degree ordering of `basis` if present.
"""
function approximatefekete(base_ring, basis, samples; show_det=false, s=3)
    V, P, samples = approximate_fekete(samples, basis, show_det=show_det , s=s, prec=precision(base_ring))
    R = SampledMPolyRing(base_ring, samples)
    [SampledMPolyRingElem(R, base_ring.(V[:,p]))  for p in eachindex(basis)], R.samples
end

###################################
### Low rank matrix polynomials ###
###################################

"""
    LowRankMatPol(eigenvalues::Vector, rightevs::Vector{Vector}[, leftevs::Vector{Vector}])

The matrix ``∑_i λ_i v_i w_i^T``, where v is specified by rightevs and w by leftevs

If `leftevs` is not specified, use `leftevs = rightevs`.
"""
struct LowRankMatPol
    eigenvalues::Vector
    leftevs::Vector{Vector}
    rightevs::Vector{Vector}
    function LowRankMatPol(eigenvalues, leftevs, rightevs)
        if !(length(eigenvalues) == length(leftevs) == length(rightevs))
            error("LowRankMatPol should have the same number of values as vectors")
        end
        new(eigenvalues, leftevs, rightevs)
    end
end

function LowRankMatPol(eigenvalues, eigenvectors)
    LowRankMatPol(eigenvalues, eigenvectors, eigenvectors)
end

function Base.map(f, p::LowRankMatPol)
    LowRankMatPol(f.(p.eigenvalues),[f.(v) for v in p.leftevs], [f.(v) for v in p.rightevs])
end

LinearAlgebra.transpose(A::LowRankMatPol) = LowRankMatPol(A.eigenvalues, A.rightevs, A.leftevs)

Base.getindex(A::LowRankMatPol, i, j) = sum(A.eigenvalues[r] * A.leftevs[r][j] * A.rightevs[r][i] for r=1:length(A.eigenvalues))

function Base.size(m::LowRankMatPol, i)
    length(m.leftevs[1])
end
function Base.size(m::LowRankMatPol)
    (length(m.rightevs[1]),length(m.leftevs[1]))
end

function Base.Matrix(p::LowRankMatPol)
    sum(p.eigenvalues[i] .* (p.rightevs[i] * transpose(p.leftevs[i])) for i in eachindex(p.eigenvalues))
end

#################################
### Sample evaluate functions ###
#################################

function Arblib.Arb(q::FieldElem; prec=precision(BigFloat))
    setprecision(BigFloat, prec) do
        Arb(BigFloat(q), prec=prec)
    end
end

# function Arblib.ArbRef(q::FieldElem; prec=precision(BigFloat))
#     setprecision(BigFloat, prec) do
#         Arb(BigFloat(q), prec=prec)
#     end
# end

function Arblib.Arb(q::RingElem; prec=precision(BigFloat))
    setprecision(BigFloat, prec) do
        Arb(BigFloat(q), prec=prec)
    end
end

# function Arblib.ArbRef(q::RingElem; prec=precision(BigFloat))
#     setprecision(BigFloat, prec) do
#         Arb(BigFloat(q), prec=prec)
#     end
# end

function Arblib.ArbRefMatrix(m::Matrix; prec=precision(BigFloat))
    nm = ArbRefMatrix(size(m, 1), size(m, 2), prec=prec)
    for i = 1:size(m, 1), j = 1:size(m, 2)
        nm[i, j] = Arb(m[i, j], prec=prec)
    end
    nm
end

function Arblib.ArbRefMatrix(m::Vector; prec=precision(BigFloat))
    nm = ArbRefMatrix(length(m), 1, prec=prec)
    for i in eachindex(m)
        nm[i, 1] = Arb(m[i], prec=prec)
    end
    nm
end

Nemo.evaluate(p::Int, s) = p

function Nemo.evaluate(p::LowRankMatPol, sample; scaling=1)
    LowRankMatPol([scaling*evaluate(v, sample) for v in p.eigenvalues],
               [[evaluate(v, sample) for v in w] for w in p.leftevs],
               [[evaluate(v, sample) for v in w] for w in p.rightevs])
end

function Nemo.evaluate(p::Matrix, sample::Vector{T}; scaling=1) where T
    res = [evaluate(p[i, j], sample) for i=1:size(p, 1), j=1:size(p, 2)]
    scaling * res
end

function sampleevaluate(p::LowRankMatPol, sample; scaling=1, prec=precision(BigFloat))
    LowRankMat([sampleevaluate(v, sample, scaling=scaling, prec=prec) for v in p.eigenvalues],
               [ArbRefMatrix([sampleevaluate(v, sample, prec=prec) for v in w], prec=prec) for w in p.leftevs],
               [ArbRefMatrix([sampleevaluate(v, sample, prec=prec) for v in w], prec=prec) for w in p.rightevs])
end

function sampleevaluate(p::LowRankMatPol, sample...; scaling=1, prec=precision(BigFloat))
    Arb(scaling, prec=prec) * sampleevaluate(p, collect(sample), prec=prec)
end

#evaluation for normal matrices
function sampleevaluate(p::Matrix, sample; scaling=1, prec=precision(BigFloat))
    res = ArbRefMatrix(size(p)..., prec=prec)
    for i in eachindex(p)
        res[i] = sampleevaluate(p[i], sample,prec=prec)
    end
    Arb(scaling, prec=prec) * res
end

#Extend evaluation to numbers, whatever the sample may be
function sampleevaluate(x::T, sample; scaling=1, prec=precision(BigFloat)) where T <: Real
   Arb(scaling*x, prec=prec)
end

function sampleevaluate(p::SampledMPolyRingElem, sample; scaling=1, prec=precision(BigFloat))
    scaling * Arb(p(sample...),prec=prec)
end

function sampleevaluate(p::MatrixElem, sample; scaling=1, prec=precision(BigFloat))
    sampleevaluate(Matrix(p), sample, scaling=scaling, prec=prec)
end

function sampleevaluate(p::MPolyRingElem, sample; scaling=1, prec=precision(BigFloat))
    tot = Arb(0, prec=prec)
    for (ev, c) in zip(exponent_vectors(p), coefficients(p))
        #in case the polynomial ring and the samples are incompatible
        tot += Arb(c; prec=prec) * Arb(prod(sample .^ ev), prec=prec)
    end
    Arb(scaling, prec=prec) * tot
end

sampleevaluate(x::RingElem, s; scaling=1, prec=precision(BigFloat)) = Arb(scaling*x, prec=prec)

sampleevaluate(x::FieldElem, s; scaling=1, prec=precision(BigFloat)) = Arb(scaling*x, prec=prec)

################
### Problems ###
################

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

function name(b::Block)
    return b.l
end

function Base.isless(a::Block, b::Block)
    isless((a.l, a.r, a.s), (b.l, b.r, b.s))
end

"""
    Constraint(constant, matrixcoeff, freecoeff, samples[, scalings])

Models a polynomial constaint of the form
```math
    c(x) = ∑_l ⟨ Y^l_{r,s}, A^l_{r,s}(x) ⟩ + ∑_i y_i B_i(x)
```
with samples, where `r,s` are defined by the `Block` structure with `l` as first argument

Arguments:
  - `constant::MPolyRingElem`
  - `matrixcoeff::Dict{Block, LowRankMatPol}`
  - `freecoeff::Dict{Any, MPolyRingElem}`
  - `samples::Vector{Vector}`
  - `scalings::Vector`
"""
struct Constraint
    constant
    matrixcoeff::Dict{Block, Any}
    freecoeff::Dict
    samples::Vector{Vector}
    scalings::Vector
end

Constraint(constant, matrixcoeff, freecoeff, samples) = Constraint(constant, matrixcoeff, freecoeff, samples, [1 for s in samples])

Constraint(constant, matrixcoeff, freecoeff) = Constraint(constant, matrixcoeff, freecoeff, [[0]], [1]) #assume numbers instead of polynomials

"""
    Objective(constant, matrixcoeff::Dict{Block, Matrix}, freecoeff::Dict)

The objective for the Problem
"""
struct Objective
    constant
    matrixcoeff::Dict{Block, Matrix}
    freecoeff::Dict
end

# function objective(objective::Objective, sol::DualSolution)
#     obj = objective.constant
#     for (k, v) in program.objective
# end

"""
    Problem(maximize, objective, constraints)

Combine the objective and constraints into a low-rank polynomial problem
"""
struct Problem
    maximize::Bool
    objective::Objective
    constraints::Vector{Constraint}
end

function Base.map(f, l::Problem)
    objective = Objective(f(l.objective.constant), Dict(k => f.(v) for (k, v) in l.objective.matrixcoeff), Dict(k => f(v) for (k, v) in l.objective.freecoeff))
    constraints = []
    for c in l.constraints
        push!(constraints, Constraint(f(c.constant), Dict(k => map(f, v) for (k, v) in c.matrixcoeff), Dict(k => f(v) for (k, v) in c.freecoeff), c.samples, c.scalings))
    end
    Problem(l.maximize, objective, constraints)
end

function addconstraint!(problem::Problem, constraint::Constraint)
    push!(problem.constraints, constraint)
end

function model_psd_variables_as_free_variables!(problem::Problem, as_free)
    for l in as_free
        #For each constraint, for each Block b with l==b.l, we need to
        #   1) add the entries of the LowRankMatPol to the freecoeff dict with key b,i,j
        #   2) remove the block from the constraint (from the dict matrixcoeff)
        #For each block we use, we need to:
        #   1) add a new constraint equating the free variable with the entry of the block
        m = 0
        n = 0
        for constraint in problem.constraints
            for block in collect(keys(constraint.matrixcoeff))
                if block.l == l #we do this l now
                    if n == 0
                        n = size(constraint.matrixcoeff[block],1)
                    elseif n != size(constraint.matrixcoeff[block],1)
                        error("The blocks are not of equal sizes")
                    end
                    for i = 1:size(constraint.matrixcoeff[block],1), j = 1:size(constraint.matrixcoeff[block],2)
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
                    m = max(block.r,block.s,m)
                    delete!(constraint.matrixcoeff,block)
                end
            end
        end


        # add equality constraints for the new free variables and the new blocks
        # R, (x,) = polynomial_ring(RealField,["x"])
        for i=1:n*m
            for j=1:i
                if i==j
                    push!(problem.constraints,Constraint(0,
                        Dict(Block(l,i,j)=> hcat([1])), # matrix variable
                        Dict((l,i,j)=>-1), # free variable
                        ))
                else # i != j, so both sides get 1/2
                    # push!(sos.constraints,Constraint(0,
                    #     Dict(Block(l,i,j) => LowRankMatPol([1],[[1]]), Block(l,j,i) => LowRankMatPol([1],[[1]])),
                    #     Dict((l,i,j)=>-2),
                    # ))
                    push!(problem.constraints,Constraint(0,
                        Dict(Block(l,i,j) => hcat([1]), Block(l,j,i) => hcat([1])),
                        Dict((l,i,j)=>-2),
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
        for block in collect(keys(problem.objective.matrixcoeff)) # now this is not modified during the loop
            if block.l == l && block.r >= block.s # we only do the lower half, and we make it symmetric
                for i=1:n
                    for j=1:(block.r == block.s ? i : n)
                        if (block.r-1)*n+i == (block.s-1)*n + j # on the diagonal
                            new_blocks[Block(l, (block.r-1)*n+i, (block.s-1)*n+j)] = hcat(problem.objective.matrixcoeff[block][i,j])
                        else #not on the diagonal of the full matrix, so we need to take both sides into account
                            new_blocks[Block(l, (block.r-1)*n+i, (block.s-1)*n+j)] = hcat(problem.objective.matrixcoeff[block][i,j])
                            new_blocks[Block(l, (block.s-1)*n+j, (block.r-1)*n+i)] = hcat(problem.objective.matrixcoeff[block][i,j])
                        end
                    end
                end
            end
            if block.l == l
                delete!(problem.objective.matrixcoeff,block)
            end
        end
        for (block, value) in new_blocks
            problem.objective.matrixcoeff[block] = value
        end
    end
end


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
function Base.size(m::LowRankMat)
    (length(m.rightevs[1]), length(m.leftevs[1]))
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
function convert_to_prec(A::ArbRefMatrix,prec=precision(BigFloat))
    res = ArbRefMatrix(A,prec=prec)
    Arblib.get_mid!(res,res)
    return res
end

LinearAlgebra.transpose(A::LowRankMat) = LowRankMat(A.eigenvalues, A.rightevs, A.leftevs)

function Base.Matrix(p::LowRankMat)
    sum(p.eigenvalues[i] .* (p.rightevs[i] * transpose(p.leftevs[i])) for i in eachindex(p.eigenvalues))
end


##############################
### Clustered low rank SDP ###
##############################

struct ClusteredLowRankSDP
    maximize::Bool
    constant::Arb #constant is part of the objective, so maybe we should have constant, C, b, A, B, c or something like that.
    A::Vector{Vector{Matrix{Dict{Int,Union{LowRankMat, ArbRefMatrix}}}}}         # A[j][l][r,s][p]
    B::Vector{ArbRefMatrix}                                 # B[j]
    c::Vector{ArbRefMatrix}                                 # c[j]
    C::BlockDiagonal{ArbRef,BlockDiagonal{ArbRef,ArbRefMatrix}} # C[j][l]
    b::ArbRefMatrix                                         #b
    matrix_coeff_names::Vector{Vector{Any}}                 # the (user defined) names of the blocks
    free_coeff_names::Vector{Any}                           # the (user defined) names of the free variables
end

function ClusteredLowRankSDP(maximize, constant, A, B, c, C, b)
    matrixcoeff_names = [[(j,l) for l in eachindex(A[j])] for j in eachindex(A)]
    freecoeff_names = collect(1:size(b,1))
    ClusteredLowRankSDP(maximize, constant, A, B, c, C, b, matrixcoeff_names, freecoeff_names)
end

"""
    ClusteredLowRankSDP(sos::Problem; as_free::Vector, prec, verbose])

Define a ClusteredLowRankSDP based on the Problem sos.

The PSD variables defined by the keys in the vector `as_free` will be modelled as extra free variables,
with extra constraints to ensure that they are equal to the entries of the PSD variables.
Remaining keyword arguments:
  - `prec` (default: precision(BigFloat)) - the precision of the result
  - `verbose` (default: false) -  print progress to the standard output
"""
function ClusteredLowRankSDP(sos::Problem; prec=precision(BigFloat), verbose=false)
    verbose && println("Forming the clusters...")

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
    # sort clusters, to have a consistent order over multiple runs
    sort!(clusters, by = x->(length(x), hash(x)))

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
        @warn "Constraints which do not use SDP variables detected."
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
        highranks = Dict{Any, Bool}()
        for constraintindex in constraintindices
            constraint = sos.constraints[constraintindex]
            for block in keys(constraint.matrixcoeff)
                subblocksizes[block.l] = max(size(constraint.matrixcoeff[block], 1),get(subblocksizes,block.l,0))
                nsubblocks[block.l] = max(block.r, block.s, get(nsubblocks, block.l, 0))
                highranks[block.l] = get(highranks, block.l, false) || typeof(constraint.matrixcoeff[block]) != LowRankMatPol
            end
        end
        v = [[Dict{Int,Union{LowRankMat, ArbRefMatrix}}() for _=1:m, _=1:m] for (k, m) in nsubblocks]
		subblock_keys = collect(keys(nsubblocks)) #fix an order
        subblock_keys_dict = Dict(k=>i for (i,k) in enumerate(subblock_keys))

        i = 1
        #This can take quite long for a large amount of samples & a large basis
        for constraintindex in constraintindices
            constraint = sos.constraints[constraintindex]
            for (idx,sample) in enumerate(constraint.samples)
                for block in keys(constraint.matrixcoeff)
                    #we can make a dict from subblock_keys to indices?
                    v[subblock_keys_dict[block.l]][block.r,block.s][i] = sampleevaluate(constraint.matrixcoeff[block], sample,scaling = constraint.scalings[idx],prec=prec)
                    # We can set the s,r block here to the transpose of the r,s block. Or to 1/2 * the transpose if we would want that.
                end
                for l in keys(highranks)
                    vidx = subblock_keys_dict[l]
                    if highranks[l] 
                        all_blocks = ArbRefMatrix[get(v[vidx][r,s], i, ArbRefMatrix(zeros(subblocksizes[l],subblocksizes[l]),prec=prec)) for r=1:nsubblocks[l] for s=1:nsubblocks[l]]
                        v[vidx][1,1][i] = hvcat(nsubblocks[l], all_blocks...)
                    end
                end
                i += 1
            end
        end
        #contract high-rank blocks (only (r,s)=(1,1) is used)
        for l in keys(highranks)
            if highranks[l]
                vidx = subblock_keys_dict[l]
                v[vidx] = [v[vidx][1,1] for _=1:1, _=1:1]
            end
        end

        push!(A, v)

        M = [[ArbRefMatrix(subblocksizes[l], subblocksizes[l],prec=prec) for r=1:m, s=1:m] for (l, m) in nsubblocks]
        for block in keys(sos.objective.matrixcoeff)
            if block.l in clusters[clusterindex]
                M[subblock_keys_dict[block.l]][block.r, block.s] = ArbRefMatrix(sos.objective.matrixcoeff[block], prec=prec)
            end
        end
        push!(C, [ArbRefMatrix(hvcat(size(S,2), [S[r,s] for r=1:size(S,1) for s=1:size(S,2)]...),prec=prec) for S in M])
        push!(matrix_coeff_names, subblock_keys)
    end
    C_block = BlockDiagonal([BlockDiagonal([b for b in blocks]) for blocks in C])

    verbose && println("Sampling the coefficients of the free variables...")

    freecoefflabels = []
    for constraint in sos.constraints
        #We want an ordering for B and b; consistent over multiple runs
        append!(freecoefflabels, sort(collect(keys(constraint.freecoeff)), by=hash))
    end
    unique!(freecoefflabels) 
    freecoefflabels_dict = Dict(k=>i for (i,k) in enumerate(freecoefflabels))

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
                    constraint_B[s_idx,freecoefflabels_dict[k]] = Arb(scaling*sampleevaluate(v,s), prec=prec)
                end
            end
            push!(Bj,constraint_B)
        end
        push!(B,vcat(Bj...))
    end
    verbose && println("Sampling the constants...")
    c = [ArbRefMatrix([sampleevaluate(sos.constraints[constraintindex].scalings[s_idx]*sos.constraints[constraintindex].constant,sample) for constraintindex in constraintindices for (s_idx,sample) in enumerate(sos.constraints[constraintindex].samples)],prec=prec) for constraintindices in cluster_constraint_index]

    b = ArbRefMatrix(length(freecoefflabels),1,prec=prec)
    for (k,v) in sos.objective.freecoeff
        b[freecoefflabels_dict[k]] = Arb(v, prec=prec)
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

Base.show(io::IO, x::Optimal) = @printf(io, "pdOpt")
Base.show(io::IO, x::NearOptimal) = @printf(io,"NearOpt")
Base.show(io::IO, x::Feasible) = @printf(io,"pdFeas")
Base.show(io::IO, x::PrimalFeasible) = @printf(io,"pFeas")
Base.show(io::IO, x::DualFeasible) = @printf(io,"dFeas")
Base.show(io::IO, x::Status) = @printf(io,"NOINFO")

struct PrimalSolution{T}
    base_ring
    x::Vector{T}
    matrixvars::Dict{Block,Matrix{T}} 
end

struct DualSolution{T}
    base_ring
    matrixvars::Dict{Block,Matrix{T}}
    freevars::Dict{Any,T}
end

function objvalue(objective::Objective, sol::DualSolution{T}) where T
    obj = T(objective.constant)
    for k in keys(objective.matrixcoeff)
        obj += dot(T.(objective.matrixcoeff[k]), sol.matrixvars[k])
    end
    for k in keys(objective.freecoeff)
        obj += T.(objective.freecoeff[k]) * sol.freevars[k]
    end
    obj
end

function objvalue(problem::Problem, sol::DualSolution)
    objvalue(problem.objective, sol)
end

struct Solution{T}
    primalsol::PrimalSolution{T}
    dualsol::DualSolution{T}
end

function matrixvar(sol::DualSolution, name)
    sol.matrixvars[name]
end

function matrixvar(sol::PrimalSolution, name)
    sol.matrixvars[name]
end

function freevar(sol::DualSolution, name)
    sol.freevars[name]
end

function traceinner(m::LowRankMatPol, v::Matrix)
    # Tr(m^T * v)= Tr((rightev * leftev^T)^T * v) = dot(rightev, v * leftev)
    sum(m.eigenvalues[i] * dot(m.rightevs[i], v * m.leftevs[i]) for i in eachindex(m.eigenvalues))
end
traceinner(v::Matrix, m::LowRankMatPol) = traceinner(m, v)

LinearAlgebra.dot(v::Matrix, m::LowRankMatPol) = traceinner(m, v)

function traceinner(m::Matrix, v::Matrix)
    sum(2^(i!=j) * m[i, j] * v[i, j] for i=1:size(m,1) for j=i:size(m,2))
end

#compute Ax-b, where the constraints should satisfy Ax = b
function slacks(problem::Problem, sol::DualSolution)
    slacks = []
    for constraint in problem.constraints
        slack = -constraint.constant
        for (b, m) in constraint.matrixcoeff
            v = matrixvar(sol, b)
            slack += traceinner(m, v)
        end
        for (b, m) in constraint.freecoeff
            slack += m * freevar(sol, b)
        end
        push!(slacks, slack)
    end
    slacks
end

function vectorize(sol::DualSolution{T}) where T
    v = T[]
    for k in sort(collect(keys(sol.matrixvars)), by=k->(size(sol.matrixvars[k],1), hash(k)))
        m = sol.matrixvars[k]
        for i=1:size(m, 1), j=i:size(m, 2)
            push!(v, m[i, j])
        end
    end
    for k in sort(collect(keys(sol.freevars)), by=hash)
        push!(v, sol.freevars[k])
    end
    v
end

function as_dual_solution(sol::DualSolution, x::Vector{T}) where T
    t = 1

    matrixvars = Dict{Block,Matrix{T}}()
    for k in sort(collect(keys(sol.matrixvars)), by=k->(size(sol.matrixvars[k],1), hash(k)))
        v = sol.matrixvars[k]
        m = Matrix{T}(undef, size(v, 1), size(v, 1))
        for i=1:size(m, 1), j=i:size(m, 2)
            m[i, j] = m[j, i] = x[t]
            t += 1
        end
        matrixvars[k] = m
    end

    freevars = Dict{Any,T}()
    for k in sort(collect(keys(sol.freevars)), by=hash)
        freevars[k] = x[t]
        t += 1
    end

    DualSolution{T}(parent(x[1]), matrixvars, freevars)
end

function blocksizes(problem::Problem)
    mvd = Dict()
    for constraint in problem.constraints, (k, v) in constraint.matrixcoeff
        mvd[k] = size(v, 1)
    end
	mvd
end



"""
	partial_linearsystem(problem::Problem, sol::DualSolution, columns::Union{DualSolution, Vector{Int}})

Let x be the vcat of the vectorizations of the matrix variables. Let I be the index set of the columns.
This function returns the matrix A_I and vector b-Ax such that the constraints for 
the error vector e using variables indexed by I are given by the system A_Ie = b-Ax.
"""
function partial_linearsystem(problem::Problem, sol::DualSolution, columns::Vector{Int}; FF=QQ, monomial_bases=nothing, verbose=false)
    # create a dual solution with 1 if the variable should be in the system and 0 otherwise
    x_cols = zero(vectorize(sol))
    for i in columns
        x_cols[i] = 1
    end
    columns_sol = as_dual_solution(sol, x_cols)
    # create a vector with indices being the columns that need to be taken into account
    # x_cols = vectorize(columns)
    # columns = [i for i in eachindex(x_cols) if x_cols[i] != 0]
    A,b = partial_linearsystem(problem, sol, columns_sol; FF=FF, monomial_bases=monomial_bases, verbose=verbose)
    # get the correct order. A has now the order of sort(columns)
    p = sortperm(columns)
    A[:, invperm(p)], b
end
function partial_linearsystem(problem::Problem, sol::DualSolution, columns::DualSolution; FF=QQ, monomial_bases=nothing, verbose=false)
    verbose && print("Creating the right hand side b-Ax...")
    t = @elapsed begin
        rhs = -slacks(problem, sol) # b - Ax
        if isnothing(monomial_bases)
            b = [evaluate(rhs[i], s) for i in eachindex(rhs) for s in problem.constraints[i].samples]
            b = matrix(FF, length(b), 1, b)
        else
            Rs = [parent(last(m)) for m in monomial_bases]
            exp_vecs = [[exponent_vector(m,1) for m in mons] for mons in monomial_bases]
            mon_lengths = [sum(length.(monomial_bases)[1:k-1]; init=0) for k=1:length(monomial_bases)]
            indices_expvecs = [Dict(exp_vecs[k][i]=>mon_lengths[k] + i for i in eachindex(exp_vecs[k])) for k in eachindex(exp_vecs)]
            b = zero_matrix(FF, sum(length.(monomial_bases)), 1)
            for k in eachindex(rhs) #constraints
                for (c,ev) in zip(coefficients(Rs[k](rhs[k])), exponent_vectors(Rs[k](rhs[k])))
                    i = get(indices_expvecs[k], ev, nothing)
                    if !isnothing(i)
                        b[i, 1] = c
                    end
                end
            end
        end
    end
    verbose && println(" $t")

    nrs = sum(length(constraint.samples) for constraint in problem.constraints)

	mvd = blocksizes(problem)
    free_vars = sort(unique([k for constraint in problem.constraints for (k, v) in constraint.freecoeff]), by=hash)
    # nf = length(free_vars)
	# ncs = sum(binomial(s+1, 2) for s in values(mvd)) + nf
    ncs = sum(!=(0), vectorize(columns))

    # The T doesn't work when the first values are, e.g., over QQ and the latter over Q(sqrt(2))
	# T = typeof(evaluate(first(values(problem.constraints[1].matrixcoeff))[1,1], problem.constraints[1].samples[1]))
    # Q: how can we do this better so that we return a Matrix & Vector with type?
    A = zero_matrix(FF, nrs, ncs)#[FF(0) for i=1:nrs, j=1:ncs]
	# b = zero_matrix(FF, nrs, 1) #[FF(0) for i=1:nrs]

    j = 1 # the column index; only increased when we encounter entries we consider
    #this is the block we now consider
    verbose && print("Creating the matrix A of the partial linear system...")
    t = @elapsed begin
        for bln in sort(collect(keys(mvd)), by=x->(mvd[x], hash(x)))
            s = mvd[bln] # the size of the block
            for a=1:s, b=a:s
                if columns.matrixvars[bln][a,b] != 0
                    i = 1 # the constraint index (for sampling)
                    for (k, constraint) in enumerate(problem.constraints)
                        if haskey(constraint.matrixcoeff, bln) #the block is used in this constraint
                            v = constraint.matrixcoeff[bln][a,b]
                            # fill the part of the column corresponding to this constraint
                            if isnothing(monomial_bases)
                                #evaluate on samples
                                for sample in constraint.samples
                                    A[i,j] = 2^(a != b) * evaluate(v, sample)
                                    i+=1
                                end
                            else
                                # coefficient matching
                                for (c,ev) in zip(coefficients(Rs[k](v)), exponent_vectors(Rs[k](v)))
                                    i = get(indices_expvecs[k], ev, nothing)
                                    if !isnothing(i)
                                        A[i, j] = 2^(a != b) * c
                                    end
                                end
                            end
                        elseif isnothing(monomial_bases) 
                            # with coefficient matching we don't have to keep track of which constraint we are doing right now
                            i += length(constraint.samples)
                        end
                    end
                    j += 1 # next column
                end
            end
        end
        # free vars
        for f in free_vars
            if columns.freevars[f] != 0 # we consider this entry
                i = 1
                for (k, constraint) in enumerate(problem.constraints)
                    if haskey(constraint.freecoeff, f)
                        if isnothing(monomial_bases)
                            # sampling
                            for sample in constraint.samples
                                A[i, j] = evaluate(constraint.freecoeff[f], sample)
                                i += 1
                            end
                        else
                            #coefficient matching
                            for (c,ev) in zip(coefficients(Rs[k](constraint.freecoeff[f])), exponent_vectors(Rs[k](constraint.freecoeff[f])))
                                i = get(indices_expvecs[k], ev, nothing)
                                if !isnothing(i)
                                    A[i,j] = c
                                end
                            end
                        end
                    else
                        i += length(constraint.samples)
                    end
                end
                j += 1
            end
        end
    end
    verbose && println(" $t")
    return A, b
end



"""
	linearsystem(problem::Problem)

Let x be the vcat of the vectorizations of the matrix variables.
This function returns the matrix A and vector b such that the constraints are
given by the system Ax = b.
"""
function linearsystem(problem::Problem, FF=QQ)
    nrs = sum(length(constraint.samples) for constraint in problem.constraints)

	mvd = blocksizes(problem)
    free_vars = sort(unique([k for constraint in problem.constraints for (k, v) in constraint.freecoeff]), by=hash)
    nf = length(free_vars)
	ncs = sum(binomial(s+1, 2) for s in values(mvd)) + nf

    # The T doesn't work when the first values are, e.g., over QQ and the latter over Q(sqrt(2))
	# T = typeof(evaluate(first(values(problem.constraints[1].matrixcoeff))[1,1], problem.constraints[1].samples[1]))
    # Q: how can we do this better so that we return a Matrix & Vector with type?
    A = zero_matrix(FF, nrs, ncs)#[FF(0) for i=1:nrs, j=1:ncs]
	b = zero_matrix(FF, nrs, 1) #[FF(0) for i=1:nrs]

	i = 1
	for constraint in problem.constraints, sample in constraint.samples
	    j = 1
		for	bln in sort(collect(keys(mvd)), by=x->(mvd[x], hash(x)))
		    s = mvd[bln]
		    if haskey(constraint.matrixcoeff, bln)
		        block = constraint.matrixcoeff[bln]
		        eb = evaluate(block, sample)
		        for a=1:s, b=a:s
		            A[i, j] = FF(eb[a, b])
			        if a != b
			            A[i, j] *= 2
			        end
			        j += 1
			    end
			else
			    for a=1:s, b=a:s
			        A[i, j] = FF(0)
			        j += 1
			    end
			end
		end
        for f in free_vars
            if haskey(constraint.freecoeff, f)
                v = constraint.freecoeff[f]
                A[i,j] = FF(evaluate(v, sample))
                j+=1
            else
                A[i,j] = FF(0)
                j+=1
            end
        end

		b[i, 1] = FF(evaluate(constraint.constant, sample))
		i += 1
	end
	A, b
end


"""
    linearsystem_coefficientmatching(problem::Problem, monomial_basis::Vector{Vector{MPolyRingElem}}, FF=QQ)

Let x be the vcat of the vectorizations of the matrix variables.
This function returns the matrix A and vector b such that the constraints obtained 
from coefficient matching are given by the system Ax = b over the field FF, with 
one constraint per monomial in monomial_basis. The problem should not contain 
SampledMPolyRingElem's for this to work.
"""
function linearsystem_coefficientmatching(problem::Problem, monomial_bases, FF=QQ)
    Rs = [parent(last(m)) for m in monomial_bases]
    # base_rings = [base_ring(R) for R in Rs]

    nrs = sum(length(constraint.samples) for constraint in problem.constraints)
    @assert nrs == sum(length.(monomial_bases))

	mvd = blocksizes(problem)
    free_vars = sort(unique([k for constraint in problem.constraints for (k, v) in constraint.freecoeff]), by=hash)
    nf = length(free_vars)
	ncs = sum(binomial(s+1, 2) for s in values(mvd)) + nf

    # The T doesn't work when the first values are, e.g., over QQ and the latter over Q(sqrt(2))
	# T = typeof(evaluate(first(values(problem.constraints[1].matrixcoeff))[1,1], problem.constraints[1].samples[1]))
    # Q: how can we do this better so that we return a Matrix & Vector with type?
    A = zero_matrix(FF, nrs, ncs)#[FF(0) for i=1:nrs, j=1:ncs]
	b = zero_matrix(FF, nrs, 1)#[FF(0) for i=1:nrs]
    #TODO: find the monomials (now input for easyness)
    exp_vecs = [[exponent_vector(m,1) for m in mons] for mons in monomial_bases]
    
    
    mon_lengths = [sum(length.(monomial_bases)[1:k-1]; init=0) for k=1:length(monomial_bases)]
    indices_expvecs = [Dict(exp_vecs[k][i]=>mon_lengths[k] + i for i in eachindex(exp_vecs[k])) for k in eachindex(exp_vecs)]
    #constraint index: sum(length.(mon[1:k-1])) + ik
	for (k,constraint) in enumerate(problem.constraints)#, (ik, mon) in enumerate(exp_vecs[k]) 
	    jsum = 1
		for	bln in sort(collect(keys(mvd)), by=x->(mvd[x], hash(x)))
		    s = mvd[bln]
		    if haskey(constraint.matrixcoeff, bln)
		        block = constraint.matrixcoeff[bln]
                eb = [block[a,b] for a in 1:size(block,1), b in 1:size(block,2)]
                # for (ik, mon) in enumerate(monomials[k])
                #     i = mon_lengths[k] + ik
                j = jsum
                for a=1:s, b=a:s
                    for (c,ev) in zip(coefficients(Rs[k](eb[a, b])), exponent_vectors(Rs[k](eb[a, b])))
                        i = get(indices_expvecs[k], ev, nothing)
                        if !isnothing(i)
                            A[i, j] = c
                            if a != b
                                A[i, j] *= 2
                            end
                        end
                    end
                    j += 1
                end
                # end
			# else
                # for (ik, mon) in enumerate(monomials[k])
                #     i = mon_lengths[k] + ik
                #     j = jsum
                #     for a=1:s, b=a:s
                #         A[i, j] = parent(constraint.constant)(0)
                #         j += 1
                #     end
                # end
			end
            jsum += div(s*(s+1),2)
		end
        # for (ik, mon) in enumerate(monomials[k])
        # i = mon_lengths[k] + ik
        j = jsum
        for f in free_vars
            if haskey(constraint.freecoeff, f)
                for (c,ev) in zip(coefficients(Rs[k](constraint.freecoeff[f])), exponent_vectors(Rs[k](constraint.freecoeff[f])))
                    i = get(indices_expvecs[k], ev, nothing)
                    if !isnothing(i)
                        A[i,j] = c
                    end
                end
            end
            j+=1
        end
        for (c,ev) in zip(coefficients(Rs[k](constraint.constant)), exponent_vectors(Rs[k](constraint.constant)))
            i = get(indices_expvecs[k], ev, nothing)
            if !isnothing(i)
                b[i, 1] = c
            end
        end
        # end

	end
	A, b
end
