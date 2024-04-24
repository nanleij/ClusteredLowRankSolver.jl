#########################
### Missing functions ###
#########################

Base.precision(::Nemo.AbstractAlgebra.Floats{BigFloat}) = precision(BigFloat)

######################################
### Completely sampled polynomials ###
######################################

"""
    SampledMPolyRing(base_ring, samples)

A sampled polynomial ring with evaluations in `base_ring`, only defined on the `samples`.
The samples should be sorted. 
"""
struct SampledMPolyRing{T} <: Nemo.Ring
    base_ring
    samples
    function SampledMPolyRing(base_ring, samples)
        @assert issorted(samples)
        T = typeof(base_ring(1))
        new{T}(base_ring, samples)
    end
end

"""
    sampled_polynomial_ring(base_ring, samples)

Create a `SampledMPolyRing` with values in `base_ring` defined on `samples`. The samples should be sorted.
"""
sampled_polynomial_ring(base_ring, samples) = SampledMPolyRing(base_ring, samples)

==(a::SampledMPolyRing, b::SampledMPolyRing) = a.base_ring == b.base_ring && a.samples == b.samples

"""
    SampledMPolyRinElem(parent::SampledMPolyRing, evaluations::Vector)

A sampled polynomial corresponding to the given parent, which evaluates to `evaluations` on the samples of the parent ring.
"""
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

(R::SampledMPolyRing)(p::Union{QQMPolyRingElem, MPolyRingElem}) = SampledMPolyRingElem(R, [R.base_ring(p(x...)) for x in R.samples])
(R::SampledMPolyRing)(p::Union{QQPolyRingElem, PolyRingElem}) = SampledMPolyRingElem(R, [R.base_ring(p(x...)) for x in R.samples])


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

for S in [QQMPolyRingElem, ZZMPolyRingElem]
    function (p::S)(sp::Vararg{SampledMPolyRingElem{T}}) where {T <: RingElem}
        length(sp) == nvars(parent(p)) || ArgumentError("Number of variables does not match number of values")
        @assert all(q.parent == sp[1].parent for q in sp)
        r = zero(sp[1])
        for si in eachindex(r.evaluations)
            r.evaluations[si] = p([q.evaluations[si] for q in sp]...)
        end
        return r
    end
end
function (p::Generic.MPoly{T})(sp::Vararg{SampledMPolyRingElem{T}}) where {T <: RingElem}
    length(sp) == nvars(parent(p)) || ArgumentError("Number of variables does not match number of values")
    @assert all(q.parent == sp[1].parent for q in sp)
    r = zero(sp[1])
    for si in eachindex(r.evaluations)
        r.evaluations[si] = p([q.evaluations[si] for q in sp]...)
    end
    return r
end
for S in [QQPolyRingElem, ZZPolyRingElem]
    function (p::S)(sp::SampledMPolyRingElem{T}) where T <: RingElem
        r = zero(sp)
        for si in eachindex(parent(sp).samples)
            r.evaluations[si] = p(sp.evaluations[si])
        end
        return r        
    end
end
function (p::Generic.Poly{T})(sp::SampledMPolyRingElem{T}) where T <: RingElem
    r = zero(sp)
    for si in eachindex(parent(sp).samples)
        r.evaluations[si] = p(sp.evaluations[si])
    end
    return r        
end

function evaluate(p::SampledMPolyRingElem, v)
    # univariate edge case: convert to number or vector depending on the samples of parent(p)
    if !(v isa Vector) && first(parent(p).samples) isa Vector
        v = [v]
    elseif (v isa Vector) && length(v) == 1 && !(first(parent(p).samples) isa Vector)
        v = first(v)
    end
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

function Base.:+(p::T, q::SampledMPolyRingElem) where {T <: RingElem}
    parent(q)(p) + q
end
function Base.:+(q::SampledMPolyRingElem, p::T) where {T <: RingElem}
    p + q
end


# subtraction
function Base.:-(q::SampledMPolyRingElem)
    SampledMPolyRingElem(parent(q),-q.evaluations)
end

function Base.:-(p::Union{PolyRingElem, MPolyRingElem}, q::SampledMPolyRingElem) 
    p + (-q)
end
function Base.:-(p::SampledMPolyRingElem, q::SampledMPolyRingElem)
    p + (-q)
end

function Base.:-(p::SampledMPolyRingElem, q::Union{PolyRingElem, MPolyRingElem})
    p + (-q)
end

function Base.:-(p::T, q::SampledMPolyRingElem) where {T <: RingElem}
    parent(q)(p) - q
end
function Base.:-(q::SampledMPolyRingElem, p::T) where {T <: RingElem}
    (-p) + q
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
    approximatefekete(basis, samples; base_ring=BigFloat) -> basis, samples

Compute approximate fekete points based on samples and a corresponding orthogonal basis.
The basis consists of sampled polynomials, sampled at `samples`.

This preserves a degree ordering of `basis` if present.
"""
function approximatefekete(basis, samples; base_ring=BigFloat, show_det=false, s=3, verbose=false)
    V, P, samples = approximate_fekete(samples, basis, show_det=show_det , s=s, prec=precision(base_ring), verbose=verbose)
    R = SampledMPolyRing(base_ring, samples)
    [SampledMPolyRingElem(R, base_ring.(V[:,p]))  for p in eachindex(basis)], R.samples
end

###################################
### Low rank matrix polynomials ###
###################################

"""
    LowRankMatPol(lambda::Vector, vs::Vector{Vector}[, ws::Vector{Vector}])

The matrix ``∑_i λ_i v_i w_i^{\\sf T}`` where ``v_i`` are the entries of `vs` and ``w_i`` the entries of `ws`

If `ws` is not specified, use `ws = vs`.
"""
struct LowRankMatPol
    lambda::Vector
    vs::Vector{Vector}
    ws::Vector{Vector}
    function LowRankMatPol(lambda, vs, ws)
        if !(length(lambda) == length(ws) == length(vs))
            error("LowRankMatPol should have the same number of values as vectors")
        end
        if length(unique(length.(vs))) != 1 || length(unique(length.(ws))) != 1
            error("The size of the rank 1 factors in LowRankMatPol should be consistent")
        end
        new(lambda, vs, ws)
    end
end

function LowRankMatPol(lambda, vs)
    LowRankMatPol(lambda, vs, vs)
end

function Base.map(f, p::LowRankMatPol)
    LowRankMatPol(f.(p.lambda), [f.(x) for x in p.vs], [f.(x) for x in p.ws])
end

LinearAlgebra.transpose(A::LowRankMatPol) = LowRankMatPol(A.lambda, A.ws, A.vs)

Base.getindex(A::LowRankMatPol, i, j) = sum(A.lambda[r] * A.ws[r][j] * A.vs[r][i] for r=1:length(A.lambda))

function Base.size(m::LowRankMatPol, i)
    @assert 1 <= i <= 2
    size(m)[i]
end
function Base.size(m::LowRankMatPol)
    (length(m.vs[1]), length(m.ws[1]))
end

function Base.Matrix(p::LowRankMatPol)
    sum(p.lambda[i] .* (p.vs[i] * transpose(p.ws[i])) for i in eachindex(p.lambda))
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

function Nemo.evaluate(p::LowRankMatPol, sample; scaling=1)
    LowRankMatPol([scaling*myevaluate(v, sample) for v in p.lambda],
               [[myevaluate(x, sample) for x in y] for y in p.vs],
               [[myevaluate(x, sample) for x in y] for y in p.ws])
end

(p::LowRankMatPol)(sample) = evaluate(p, sample)

myevaluate(x::T, s) where T <: Real = x 
myevaluate(x::T, s) where T <: RingElem = x 
myevaluate(x::T, s) where T <: Union{MPolyRingElem, PolyRingElem} = evaluate(x, s)
myevaluate(x::SampledMPolyRingElem, s) = evaluate(x, s)
myevaluate(x::LowRankMatPol, s) = evaluate(x, s)
function myevaluate(p::Matrix, sample; scaling=1)
    res = [myevaluate(p[i, j], sample) for i=1:size(p, 1), j=1:size(p, 2)]
    scaling * res
end

function sampleevaluate(p::LowRankMatPol, sample; scaling=1, prec=precision(BigFloat))
    LowRankMat([sampleevaluate(v, sample, scaling=scaling, prec=prec) for v in p.lambda],
               [ArbRefMatrix([sampleevaluate(v, sample, prec=prec) for v in w], prec=prec) for w in p.ws],
               [ArbRefMatrix([sampleevaluate(v, sample, prec=prec) for v in w], prec=prec) for w in p.vs])
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

function sampleevaluate(p::Union{QQPolyRingElem, PolyRingElem}, sample; scaling=1, prec=precision(BigFloat))
    tot = Arb(0, prec=prec)
    for (ev, c) in zip(exponent_vectors(p), coefficients(p))
        #in case the polynomial ring and the samples are incompatible
        power = first(ev)
        tot += Arb(c; prec=prec) * Arb(prod(sample ^ power), prec=prec)
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

Specifying `r,s` makes the `Block` correspond to the `r,s` subblock of the variable `l`.
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
name(b) = b
function subblock(b::Block)
    return (b.r, b.s)
end
subblock(b) = (1,1)

function Base.isless(a::Block, b::Block)
    isless((a.l, a.r, a.s), (b.l, b.r, b.s))
end
function Base.isless(a::Block, b::Any)
    isless((name(a), subblock(a)...), (name(b), subblock(b)...))
end
function Base.isless(a::Any, b::Block)
    isless((name(a), subblock(a)...), (name(b), subblock(b)...))
end

"""
    Constraint(constant, matrixcoeff, freecoeff[, samples, scalings])

Models a polynomial constaint of the form
```math
    ∑_i ⟨ A_i(x), Y_i  ⟩ + ∑_j b_j(x) y_j = c(x)
```
using sampling on `samples`. Here the samples are only required if ``A_i``, ``b_j``, and ``c`` are polynomials.
When the coefficient matrix ``A_i`` has block structure with equal sized blocks, the
`Block` struct can be used as key to indicate to which subblock the given matrix corresponds.

Arguments:
  - `constant`: The right hand side ``c(x)``
  - `matrixcoeff::Dict`: The coefficient matrices for the positive semidefinite matrix variables.
  - `freecoeff::Dict`: The coefficients for the free variables.
  - `samples::Vector`: The sample set on which the constraint should be satisfied. Required for polynomial constraints.
  - `scalings::Vector`: Optional; scale the constraint with a factor depending on the sample index.
"""
struct Constraint
    constant
    matrixcoeff::Dict
    freecoeff::Dict
    samples::Vector
    scalings::Vector
end

Constraint(constant, matrixcoeff, freecoeff, samples) = Constraint(constant, matrixcoeff, freecoeff, samples, [1 for s in samples])

Constraint(constant, matrixcoeff, freecoeff) = Constraint(constant, matrixcoeff, freecoeff, [0], [1]) #assume numbers instead of polynomials

"""
    Objective(constant, matrixcoeff::Dict, freecoeff::Dict)

The objective for the Problem.

Arguments:
  - `constant`: A constant offset of the objective value.
  - `matrixcoeff`: A `Dict` with positive semidefinite matrix variable names as keys and the objective matrices as values.
  - `freecoeff`: A `Dict` with free variable names as keys and the coefficients as values.
"""
struct Objective
    constant
    matrixcoeff::Dict
    freecoeff::Dict
end

"""
    matrixcoeff(x::Union{Constraint, Objective}, name)

Return the matrix coefficient corresponding to `name`
"""
function matrixcoeff(x::Union{Constraint, Objective}, name)
    x.matrixcoeff[name]
end
"""
    matrixcoeffs(x::Union{Constraint, Objective})

Return the dictionary of matrix coefficients
"""
function matrixcoeffs(x::Union{Constraint, Objective})
    x.matrixcoeff
end
"""
    freecoeffs(x::Union{Constraint, Objective})

Return the dictionary of coefficients for the free variables
"""
function freecoeffs(x::Union{Constraint, Objective})
    x.freecoeff
end
"""
    freecoeff(x::Union{Constraint, Objective}, name)

Return the coefficient of the free variable corresponding to `name`
"""
function freecoeff(x::Union{Constraint, Objective}, name)
    x.freecoeff[name]
end

"""
    Maximize(obj)

Maximize the objective
"""
struct Maximize
    objective::Objective
end
"""
    Minimize(obj)

Minimize the objective
"""
struct Minimize
    objective::Objective
end

"""
```
    Problem(maximize::Bool, objective::Objective, constraints::Vector{Constraint})
```
```
    Problem(obj::Union{Maximize, Minimize}, constraints::Vector)
```
Combine the objective and constraints into a low-rank polynomial problem.
"""
struct Problem
    maximize::Bool
    objective::Objective
    constraints::Vector{Constraint}
    function Problem(maximize::Bool, objective::Objective, constraints::Vector)
        @assert all(x isa Constraint for x in constraints)
        new(maximize, objective, constraints)
    end
end

function Problem(obj::Maximize, constraints::Vector)
    return Problem(true, objective(obj), constraints)
end
function Problem(obj::Minimize, constraints::Vector)
    return Problem(false, objective(obj), constraints)
end

"""
    constraints(problem::Problem)

Return the constraints of `problem`.
"""
constraints(problem::Problem) = problem.constraints

"""
    objective(x::Union{Maximize, Minimize, Problem})

Return the objective.
"""
function objective(obj::Union{Maximize, Minimize, Problem})
    obj.objective
end

"""
    map(f, problem::Problem)

Apply the function `f` to all coefficients in `problem`. 
"""
function Base.map(f, l::Problem)
    objective = Objective(f(l.objective.constant), Dict(k => f.(v) for (k, v) in l.objective.matrixcoeff), Dict(k => f(v) for (k, v) in l.objective.freecoeff))
    constraints = []
    for c in l.constraints
        push!(constraints, Constraint(f(c.constant), Dict(k => map(f, v) for (k, v) in c.matrixcoeff), Dict(k => f(v) for (k, v) in c.freecoeff), c.samples, c.scalings))
    end
    Problem(l.maximize, objective, constraints)
end

"""
    addconstraint!(problem, constraint)

Add `constraint` to the constraints of `problem`. 
"""
function addconstraint!(problem::Problem, constraint::Constraint)
    push!(problem.constraints, constraint)
end

"""
    model_psd_variables_as_free_variables!(problem::Problem, as_free)

Model the positive semidefinite variables with names in `as_free` using free variables,
 and add extra constraints to set them equal to auxilliary positive semidefinite variables.
"""
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
                if name(block) == l #we do this l now
                    if n == 0
                        n = size(constraint.matrixcoeff[block],1)
                    elseif n != size(constraint.matrixcoeff[block],1)
                        error("The blocks are not of equal sizes")
                    end
                    for i = 1:size(constraint.matrixcoeff[block],1), j = 1:size(constraint.matrixcoeff[block],2)
                        r, s = subblock(block)
                        if r == s && i >= j
                            #diagonal (r,s) block, so we only need to do things when i >= j
                            if i == j # just the entry
                                constraint.freecoeff[(l,(r-1)*n+i,(s-1)*n+j)] = constraint.matrixcoeff[block][i,j]
                            else #i > j, so we need to take both sides of the diagonal into account
                                constraint.freecoeff[(l,(r-1)*n+i,(s-1)*n+j)] = 2*constraint.matrixcoeff[block][i,j]
                            end
                        elseif r > s #the other case is r < s, but because it is symmetric we do that now too. (hence the factor 2)
                            constraint.freecoeff[(l,(r-1)*n+i,(s-1)*n+j)] = 2 * constraint.matrixcoeff[block][i,j]
                        end
                    end
                    m = max(r,s,m)
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
            r,s = subblock(block)
            if name(block) == l && r >= s # we only do the lower half, and we make it symmetric
                for i=1:n
                    for j=1:(r == block.s ? i : n)
                        if (r-1)*n+i == (s-1)*n + j # on the diagonal
                            new_blocks[Block(l, (r-1)*n+i, (s-1)*n+j)] = hcat(problem.objective.matrixcoeff[block][i,j])
                        else #not on the diagonal of the full matrix, so we need to take both sides into account
                            new_blocks[Block(l, (r-1)*n+i, (s-1)*n+j)] = hcat(problem.objective.matrixcoeff[block][i,j])
                            new_blocks[Block(l, (s-1)*n+j, (r-1)*n+i)] = hcat(problem.objective.matrixcoeff[block][i,j])
                        end
                    end
                end
            end
            if name(block) == l
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
    lambda::Vector{Arb}
    vs::Vector{ArbRefMatrix}
    ws::Vector{ArbRefMatrix}
end
function LowRankMat(lambda, vs)
    LowRankMat(lambda, vs, vs)
end

Base.getindex(A::LowRankMat, i, j) = sum(A.lambda[r] * A.vs[r][i] * A.ws[r][j] for r=1:length(A.lambda))

function Base.size(m::LowRankMat, i)
    @assert 1 <= i <= 2
    size(m)[i]
end
function Base.size(m::LowRankMat)
    (length(m.vs[1]), length(m.ws[1]))
end


function convert_to_prec(A::LowRankMat,prec=precision(BigFloat))
    evs = [Arb(ev,prec=prec) for ev in A.lambda]
    vs = [ArbRefMatrix(vec,prec=prec) for vec in A.vs]
    ws = [ArbRefMatrix(vec,prec=prec) for vec in A.ws]
    for i=1:length(evs)
        Arblib.get_mid!(evs[i],evs[i])
        Arblib.get_mid!(vs[i], vs[i])
        Arblib.get_mid!(ws[i], ws[i])
    end
    return LowRankMat(evs, vs, ws)
end
function convert_to_prec(A::ArbRefMatrix, prec=precision(BigFloat))
    res = ArbRefMatrix(A, prec=prec)
    Arblib.get_mid!(res, res)
    return res
end

LinearAlgebra.transpose(A::LowRankMat) = LowRankMat(A.lambda, A.ws, A.vs)

function Base.Matrix(p::LowRankMat)
    sum(p.lambda[i] .* (p.vs[i] * transpose(p.ws[i])) for i in eachindex(p.lambda))
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
    matrix_coeff_blocks::Vector{Vector{Tuple{Bool, Int}}}   # whether the user used Block and the number of subblocks 
end

function ClusteredLowRankSDP(maximize, constant, A, B, c, C, b)
    matrixcoeff_names = [[(j,l) for l in eachindex(A[j])] for j in eachindex(A)]
    freecoeff_names = collect(1:size(b,1))
    matrix_coeff_blocks = [[(false, 1) for l in block] for block in A]
    ClusteredLowRankSDP(maximize, constant, A, B, c, C, b, matrixcoeff_names, freecoeff_names,matrix_coeff_blocks)
end

"""
    ClusteredLowRankSDP(problem::Problem; prec=precision(BigFloat), verbose=false)

Define a `ClusteredLowRankSDP` based on `problem`.

Keyword arguments:
  - `prec` (default: `precision(BigFloat)`): the precision of the result
  - `verbose` (default: false): print progress to the standard output
"""
function ClusteredLowRankSDP(sos::Problem; prec=precision(BigFloat), verbose=false)
    verbose && println("Forming the clusters...")

    # clusters will be a vector of vectors containing block labels (so name(block))
    #NOTE: if a constraint doesn't have PSD variables, this creates an empty cluster.
    clusters = Vector{Any}[]
    for constraintindex in eachindex(sos.constraints)
        constraint = sos.constraints[constraintindex]
        clusterset = Set{Int}()
        for k in keys(constraint.matrixcoeff)
            for i in eachindex(clusters)
                if name(k) in clusters[i]
                    push!(clusterset, i)
                end
            end
        end
        clusterset = sort(collect(clusterset))
        if isempty(clusterset)
            if length(keys(constraint.matrixcoeff)) > 0 #only create a new cluster for constraints with matrix blocks
                push!(clusters, [name(k) for k in keys(constraint.matrixcoeff)])
            end
        else
            #remove the clusters which we need to combine from 'clusters', and merge them
            u = union(splice!(clusters, clusterset)...)
            append!(u, [name(k) for k in keys(constraint.matrixcoeff)])
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
            if name(k) in clusters[clusterindex]
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
    matrix_coeff_blocks = []
    matrix_coeff_names = []
    for clusterindex in eachindex(cluster_constraint_index)
        constraintindices = cluster_constraint_index[clusterindex]
        nsubblocks = Dict{Any,Int}()
        subblocksizes = Dict{Any,Int}()
        highranks = Dict{Any, Bool}()
        subblockBlock = Dict()
        for constraintindex in constraintindices
            constraint = sos.constraints[constraintindex]
            for block in keys(constraint.matrixcoeff)
                r,s = subblock(block)
                subblocksizes[name(block)] = max(size(constraint.matrixcoeff[block], 1),get(subblocksizes,name(block),0))
                nsubblocks[name(block)] = max(r, s, get(nsubblocks, name(block), 0))
                highranks[name(block)] = get(highranks, name(block), false) || typeof(constraint.matrixcoeff[block]) != LowRankMatPol
                if haskey(subblockBlock, name(block)) && subblockBlock[name(block)] != (block isa Block) 
                    println("Please use Block consistently. Solutions with Block($(name(block))) will be returned.")
                    subblockBlock[name(block)] = true
                else
                    subblockBlock[name(block)] = (block isa Block)
                end
            end
        end
        v = [[Dict{Int,Union{LowRankMat, ArbRefMatrix}}() for _=1:m, _=1:m] for (k, m) in nsubblocks]
		subblock_keys = collect(keys(nsubblocks)) #fix an order
        subblock_keys_dict = Dict(k=>i for (i,k) in enumerate(subblock_keys))
        push!(matrix_coeff_blocks, [(subblockBlock[k], nsubblocks[k]) for k in subblock_keys])

        i = 1
        #This can take quite long for a large amount of samples & a large basis
        for constraintindex in constraintindices
            constraint = sos.constraints[constraintindex]
            for (idx,sample) in enumerate(constraint.samples)
                for block in keys(constraint.matrixcoeff)
                    #we can make a dict from subblock_keys to indices?
                    r, s = subblock(block)
                    v[subblock_keys_dict[name(block)]][r,s][i] = sampleevaluate(constraint.matrixcoeff[block], sample,scaling = constraint.scalings[idx],prec=prec)
                    # We can set the s,r block here to the transpose of the r,s block. Or to 1/2 * the transpose if we would want that.
                end
                for l in keys(highranks)
                    vidx = subblock_keys_dict[l]
                    if highranks[l] 
                        all_blocks = ArbRefMatrix[get(v[vidx][r,s], i, ArbRefMatrix(zeros(subblocksizes[l],subblocksizes[l]),prec=prec)) for r=1:nsubblocks[l] for s=1:nsubblocks[l]]
                        m = hvcat(nsubblocks[l], all_blocks...)
                        if !iszero(m)
                            v[vidx][1,1][i] = m
                        end
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
            if name(block) in clusters[clusterindex]
                r, s = subblock(block)
                M[subblock_keys_dict[name(block)]][r, s] = ArbRefMatrix(sos.objective.matrixcoeff[block], prec=prec)
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
    ClusteredLowRankSDP(sos.maximize, Arb(sos.objective.constant,prec=prec), A, B, c, C_block, b, matrix_coeff_names,freecoefflabels, matrix_coeff_blocks)
end

"""
    convert_to_prec(sdp, prec=precision(BigFloat))

Convert the semidefinite program to a semidefinite program with prec bits of precision, without error bounds.
"""
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

    return ClusteredLowRankSDP(sdp.maximize,new_constant,new_A,new_B,new_c,new_C,new_b,sdp.matrix_coeff_names,sdp.free_coeff_names, sdp.matrix_coeff_blocks)
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

"""
    optimal(x::Status)

Return whether the solutions corresponding to this status are optimal.
"""
optimal(::Status) = false
optimal(::Optimal) = true

Base.show(io::IO, x::Optimal) = @printf(io, "pdOpt")
Base.show(io::IO, x::NearOptimal) = @printf(io,"NearOpt")
Base.show(io::IO, x::Feasible) = @printf(io,"pdFeas")
Base.show(io::IO, x::PrimalFeasible) = @printf(io,"pFeas")
Base.show(io::IO, x::DualFeasible) = @printf(io,"dFeas")
Base.show(io::IO, x::Status) = @printf(io,"NOINFO")

"""
    PrimalSolution{T}

A primal solution to the semidefinite program, with fields
  - `base_ring`
  - `x::Vector{T}`
  - `matrixvars::Dict{Any, Matrix{T}}`
"""
struct PrimalSolution{T}
    base_ring
    x::Vector{T}
    matrixvars::Dict{Any,Matrix{T}} 
end

"""
    DualSolution{T}

A dual solution to the semidefinite program, with fields
  - `base_ring`
  - `matrixvars::Dict{Any, Matrix{T}}`
  - `freevars::Dict{Any, T}`
"""
struct DualSolution{T}
    base_ring
    matrixvars::Dict{Any,Matrix{T}}
    freevars::Dict{Any,T}
end

"""
    objvalue(objective::Objective, sol::DualSolution)
"""
function objvalue(objective::Objective, sol::DualSolution{T}) where T <: Number
    obj = T(objective.constant)
    for k in keys(objective.matrixcoeff)
        obj += dot(T.(objective.matrixcoeff[k]), sol.matrixvars[k])
    end
    for k in keys(objective.freecoeff)
        obj += T.(objective.freecoeff[k]) * sol.freevars[k]
    end
    obj
end
# for exact fields from Nemo
function objvalue(objective::Objective, sol::DualSolution)
    obj = objective.constant
    for k in keys(objective.matrixcoeff)
        obj += dot(objective.matrixcoeff[k], sol.matrixvars[k])
    end
    for k in keys(objective.freecoeff)
        obj += objective.freecoeff[k] * sol.freevars[k]
    end
    obj
end
"""
    objvalue(problem::Problem, sol::DualSolution)

Return the objective value of the dual solution with respect to the given objective or problem.
"""
function objvalue(problem::Problem, sol::DualSolution)
    objvalue(problem.objective, sol)
end
function objvalue(obj::Union{Minimize, Maximize}, sol::DualSolution)
    objvalue(objective(obj), sol)
end

#Q: Do we want abstract struct Solution, with subtypes DualSolution and PrimalSolution?
# then stuff like matrixvar works for both
struct Solution{T}
    primalsol::PrimalSolution{T}
    dualsol::DualSolution{T}
end

"""
    matrixvar(sol, name)

Return the matrix variable corresponding to name
"""
function matrixvar(sol::DualSolution, name)
    sol.matrixvars[name]
end

function matrixvar(sol::PrimalSolution, name)
    sol.matrixvars[name]
end

"""
    matrixvars(sol)

Return the dictionary of matrix variables
"""
function matrixvars(sol::DualSolution)
    sol.matrixvars
end

function matrixvars(sol::PrimalSolution)
    sol.matrixvars
end

"""
    freevar(sol, name)

Return the free variable corresponding to name
"""
function freevar(sol::DualSolution, name)
    sol.freevars[name]
end
"""
    freevars(sol)

Return the dictionary of the free variables
"""
function freevars(sol::DualSolution)
    sol.freevars
end

function traceinner(m::LowRankMatPol, v::Matrix)
    sum(m.lambda[i] * dot(m.vs[i], v * m.ws[i]) for i in eachindex(m.lambda))
end
traceinner(v::Matrix, m::LowRankMatPol) = traceinner(m, v)

LinearAlgebra.dot(v::Matrix, m::LowRankMatPol) = traceinner(m, v)
LinearAlgebra.dot(m::LowRankMatPol, v::Matrix) = traceinner(m, v)

function traceinner(m::Matrix, v::Matrix)
    @assert size(m) == size(v)
    sum(2^(i!=j) * m[i, j] * v[i, j] for i=1:size(m,1) for j=i:size(m,2))
end

#compute Ax-b, where the constraints should satisfy Ax = b
"""
    slacks(problem, dualsol)

Compute the difference between the left hand side and the right hand side of all constraints
"""
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

"""
    vectorize(sol)

Vectorize the solution by taking the upper triangular part of the matrices. 
The variables are first sorted by size and then by hash.
"""
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

"""
    as_dual_solution(sol, x)

Undo the vectorization of x.
"""
function as_dual_solution(sol::DualSolution, x::Vector{T}) where T
    t = 1

    matrixvars = Dict{Any,Matrix{T}}()
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

#TODO: what exactly do we want here in case of subblocks?
"""
    blocksizes(problem)

Return the sizes of the matrix variables as a dictionary with the same keys
"""
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
function partial_linearsystem(problem::Problem, sol::DualSolution, columns::Vector{Int}; FF=QQ, monomial_bases=nothing)
    # create a dual solution with 1 if the variable should be in the system and 0 otherwise
    x_cols = zero(vectorize(sol))
    for i in columns
        x_cols[i] = 1
    end
    columns_sol = as_dual_solution(sol, x_cols)
    # create a vector with indices being the columns that need to be taken into account
    # x_cols = vectorize(columns)
    # columns = [i for i in eachindex(x_cols) if x_cols[i] != 0]
    A,b = partial_linearsystem(problem, sol, columns_sol; FF=FF, monomial_bases=monomial_bases)
    # get the correct order. A has now the order of sort(columns)
    p = sortperm(columns)
    A[:, invperm(p)], b
end
function partial_linearsystem(problem::Problem, sol::DualSolution, columns::DualSolution; FF=QQ, monomial_bases=nothing)
    t = @elapsed begin
        rhs = -slacks(problem, sol) # b - Ax
        if isnothing(monomial_bases)
            b = [myevaluate(rhs[i], s) for i in eachindex(rhs) for s in problem.constraints[i].samples]
            b = matrix(FF, length(b), 1, b)
        else
            Rs = [parent(last(m)) for m in monomial_bases]
            # this also works for univariate polynomials
            exp_vecs = [[last([ev for (c, ev) in zip(coefficients(m), exponent_vectors(m)) if c != 0]) for m in mons] for mons in monomial_bases]
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
                                    A[i,j] = 2^(a != b) * myevaluate(v, sample)
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
                                A[i, j] = myevaluate(constraint.freecoeff[f], sample)
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
    return A, b
end



"""
	linearsystem(problem::Problem, FF=QQ)

Let x be the vcat of the vectorizations of the matrix variables.
This function returns the matrix A and vector b such that the constraints are
given by the system Ax = b (over the field FF).
"""
function linearsystem(problem::Problem; FF=QQ, verbose = false)
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
		        eb = myevaluate(block, sample)
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
                A[i,j] = FF(myevaluate(v, sample))
                j+=1
            else
                A[i,j] = FF(0)
                j+=1
            end
        end

		b[i, 1] = FF(myevaluate(constraint.constant, sample))
		i += 1
	end
	A, b
end


"""
    linearsystem_coefficientmatching(problem::Problem, monomial_basis::Vector{Vector{MPolyRingElem}}; FF=QQ)

Let x be the vcat of the vectorizations of the matrix variables.
This function returns the matrix A and vector b such that the constraints obtained 
from coefficient matching are given by the system Ax = b over the field FF, with 
one constraint per monomial in monomial_basis. The problem should not contain 
SampledMPolyRingElem's for this to work.
"""
function linearsystem_coefficientmatching(problem::Problem, monomial_bases; FF=QQ)
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
    # This works for univariate polynomials now too due to the check on nonzero coefficients
    exp_vecs = [[last([ev for (c, ev) in zip(coefficients(m), exponent_vectors(m)) if c != 0]) for m in mons] for mons in monomial_bases]
    @assert unique.(exp_vecs) == exp_vecs
    
    
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


##############################################################################
################    Embedding a number field in R   ##########################
##############################################################################


# usage: map(x->generic_embedding(x,g), problem), where g is an approximation of the generator of the field
"""
    basic_embedding(exactvalue)

Convert the exact (rational or integer) numbers to floating point approximations.
"""
basic_embedding(x) = generic_embedding(x, BigFloat(1))

"""
    generic_embedding(exactvalue, g; base_ring=BigFloat)

Convert the exact numbers from a number field to floating point approximations, 
using a floating point approximation of a generator g of the field.

Convert rationals and integers to the same numbers in base_ring.
"""
function generic_embedding(p::MPolyRingElem, g; base_ring = RF)
    if base_ring == BigFloat
        base_ring = RF
    end
    #convert to polynomial in RealField/ArbField
    R = parent(p)
    Rnew, x = polynomial_ring(base_ring, string.(gens(R)))
    q = MPolyBuildCtx(Rnew)
    for (c, ev) in zip(coefficients(p), exponent_vectors(p))
        push_term!(q, generic_embedding(c,g, base_ring=base_ring), ev)
    end
    finish(q)
end
function generic_embedding(p::PolyRingElem, g; base_ring = RF)
    if base_ring == BigFloat
        base_ring = RF
    end
    #convert to polynomial in RealField/ArbField
    R = parent(p)
    Rnew, x = polynomial_ring(base_ring, string(gens(R)[1]))
    q = MPolyBuildCtx(Rnew)
    for (c, ev) in zip(coefficients(p), exponent_vectors(p))
        push_term!(q, generic_embedding(c,g, base_ring=base_ring), ev)
    end
    finish(q)
end

function generic_embedding(p::SampledMPolyRingElem, g; base_ring=BigFloat)
    #convert to BigFloat evaluations
    evals = generic_embedding.(p.evaluations, g, base_ring=base_ring)
    R = SampledMPolyRing(base_ring, parent(p).samples)
    return SampledMPolyRingElem(R, evals)
end

function generic_embedding(x::AbsSimpleNumFieldElem, g; base_ring=BigFloat)
    # check whether g is indeed a generator of this field
    p = defining_polynomial(parent(x))
    if base_ring == BigFloat
        @assert abs(sum(base_ring(Rational{BigInt}(coeff(p, k))) * g^k for k=0:degree(p))) < 1e-10
    end
    sum(base_ring(Rational{BigInt}(coeff(x,k))) * g^k for k=0:degree(parent(x))-1)
end

function generic_embedding(x::Union{QQFieldElem, ZZRingElem, T}, g; base_ring=BigFloat) where T <: Number
    # this is only relevant with exact problems, so numbers are either AbsSimpleNumFieldElem, or rational/int
    base_ring(Rational{BigInt}(x))
end
