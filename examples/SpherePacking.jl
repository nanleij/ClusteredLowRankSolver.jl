module SpherePacking

using ClusteredLowRankSolver, AbstractAlgebra, SpecialFunctions

export Nsphere_packing, cohnelkies

function spherevolume(n, r)
    sqrt(BigFloat(pi))^n / gamma(BigFloat(n) / 2 + 1) * r^n
end

laguerre(k, alpha, x) = basis_laguerre(k, alpha, x)[end]

function Nsphere_packing(n,d,r,N=length(r);prec=512, kwargs...)
    #The problem:
    # min M
    # M - f_ii(0)>=0 for i=1:N
    # F(f)(0) - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N >=0
    # F(f)(t) ⪰ 0 for t>=0
    # -f(w)_ij >= 0 for w>=r_i+r_j for i=1:N j=1:i (symmetric)
    # With f(x) = sum_k a_k k!/π^k L_k^{n/2-1}(π ||x||^2)
    # and thus F(f)(x) = sum_k a_k x^{k}
    # where a_k are symmetric NxN matrices
    # min M
    # s.t.  - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N + \sum_i \sum_j a_{ij,0} E_ij >= 0 (G={1}) (NxN)
    #       0 + sum_k sum_i sum_j a_{ij,k} E_{ij} x^k >= 0        (G = {1,x}) (NxN)
    #       0 - sum_k a_ijk k! pi^-k L_k^{n/2-1}(pi x) >=0          (G = {1,x - (r_i + r_j)^2}) (1x1)
    #       M - sum_k a_iik k!/pi^k L_k^{n/2-1}(0) >= 0 for i=1:N   G = {1}, 1x1
    setprecision(BigFloat,prec)
    R, x = polynomial_ring(RealField, :x)
    constraints = []

    #Constraint 1: - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N + \sum_i \sum_j a_{ij,0} E_ij is PSD. So -a_ijk + PSD_ij = -(vol B(r_i) vol B(r_j)_^1/2
    for i=1:N
        for j=1:i
            if i != j
                push!(constraints,Constraint(-sqrt(spherevolume(n,r[i])*spherevolume(n,r[j])),
                    Dict(Block(:PSD1,i,j)=>LowRankMatPol([1//2],[[1]]),
                        Block(:PSD1,j,i)=>LowRankMatPol([1//2],[[1]])),
                    Dict((0,i,j)=>-1), #we only use a_ijk for i>=j
                    ))
            else
                push!(constraints,Constraint(-sqrt(spherevolume(n,r[i])*spherevolume(n,r[j])),
                    Dict(Block(:PSD1,i,j)=>LowRankMatPol([1],[[1]])),
                    Dict((0,i,j)=>-1), #we only use a_ijk for i>=j
                    ))
            end
        end
    end

    # We orthogonalize the laguerre basis with respect to the sample points
    q = basis_laguerre(2d+1, BigFloat(n) / 2 - 1, BigFloat(2 * pi) * x)
    max_coef = [max(coefficients(q[i])...) for i = 1:length(q)]
    basis = [max_coef[i]^(-1) * q[i] for i = 1:length(q)]
    samples = sample_points_rescaled_laguerre(2d+1)
    basis, samples = approximatefekete(basis, samples)

    #constraint 2: sum_k a_{ij,k} x^k = <SOS21_ij, bb^T> + <SOS22_ij, xbb^T>
    #(Polynomial matrix constraint gives a polynomial constraint for each entry)
    for i=1:N
        for j=1:i
            PSD_part = Dict()
            free_part = Dict()
            if i != j # We have to make the stuff symmetric. The factor still matters because we use the same SOS (l) block
                for k=0:2d+1
                    free_part[(k,i,j)] = -2*x^k
                end
                PSD_part[Block(:SOS21,i,j)] = LowRankMatPol([1],[basis[1:d+1]])
                PSD_part[Block(:SOS22,i,j)] = LowRankMatPol([x],[basis[1:d+1]])
                PSD_part[Block(:SOS21,j,i)] = LowRankMatPol([1],[basis[1:d+1]])
                PSD_part[Block(:SOS22,j,i)] = LowRankMatPol([x],[basis[1:d+1]])
            else
                for k=0:2d+1
                    free_part[(k,i,j)] = -x^k
                end
                PSD_part[Block(:SOS21,i,j)] = LowRankMatPol([1],[basis[1:d+1]])
                PSD_part[Block(:SOS22,i,j)] = LowRankMatPol([x],[basis[1:d+1]])
            end
            push!(constraints,Constraint(0,PSD_part,free_part,samples))
        end
    end

    #constraint 3: SOS + (x - (r_i + r_j)^2)*SOS + sum_k a_ijk k! pi^-k L_k^{n/2-1}(pi x) = 0
    for i=1:N
        for j=1:i
            PSD_part = Dict()
            free_part = Dict()
            for k=0:2d+1
                free_part[(k,i,j)] = factorial(big(k))/BigFloat(pi)^k * laguerre(k, BigFloat(n) / 2 - 1, BigFloat(pi)*x)
            end
            #since we use 2d+1, we only need the constant of this SOS matrix; in practice it is always the 0 matrix
            PSD_part[(:SOS31,i,j)] = LowRankMatPol([1],[basis[1:1]])
            PSD_part[(:SOS32,i,j)] = LowRankMatPol([x-(r[i]+r[j])^2],[basis[1:d+1]])
            push!(constraints,Constraint(0,PSD_part,free_part,samples))
        end
    end

    #constraint 4: M - sum_k a_iik k!/pi^k L_k^{n/2-1}(0) = pos.slack
    for i=1:N
        PSD_part = Dict()
        free_part = Dict()
        for k=0:2d+1
            free_part[(k,i,i)] = factorial(big(k))/BigFloat(pi)^k * laguerre(k, BigFloat(n) / 2 - 1,0)
        end
        free_part[:M] = -1
        PSD_part[(:slack4,i)] = hcat([1])
        push!(constraints,Constraint(0,PSD_part,free_part))#,[[0]]))
    end

    #objective: M
    obj = Objective(0,Dict(),Dict(:M=>1))

    problem = Problem(Minimize(obj),constraints)
    status, primalsol, dualsol, time, errorcode = solvesdp(problem; prec=prec, kwargs...)
    return problem, primalsol, dualsol
end

function cohnelkies(n,d,r=1; prec=512, model_prec=prec,  kwargs...)
    #The problem:
    # min M
    # M - f_ii(0)>=0 for i=1:N
    # F(f)(0) - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N >=0
    # F(f)(t) >=0 for t>=0
    # -f(w)_ij >= 0 for w>=r_i+r_j for i=1:N j=1:i (symmetric)
    # With f(x) = sum_k a_k k!/pi^k L_k^{n/2-1}(pi ||x||^2)
    # and thus F(f)(x) = sum_k a_k x^{k}
    # min M
    # s.t.  - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N + \sum_i \sum_j a_{ij,0} E_ij >= 0 (G={1}) (NxN)
    #       0 + sum_k sum_i sum_j a_{ij,k} E_{ij} x^k >= 0        (G = {1,x}) (NxN)
    #       0 - sum_k a_ijk k! pi^-k L_k^{n/2-1}(pi x) >=0          (G = {1,x - (r_i + r_j)^2}) (1x1)
    #       M - sum_k a_iik k!/pi^k L_k^{n/2-1}(0) >= 0 for i=1:N   G = {1}, 1x1
    # CohnElkies.jl models the problem for 2d+1, and then one of the PSD matrices is zero. We do that now too.
    setprecision(BigFloat,model_prec)

    R, x = polynomial_ring(RealField, :x)

    q = basis_laguerre(2d+1, BigFloat(n) / 2 - 1, BigFloat(2 * pi) * x)
    max_coef = [max(coefficients(q[i])...) for i = 1:length(q)]
    basis = [max_coef[i]^(-1) * q[i] for i = 1:length(q)]
    samples = sample_points_rescaled_laguerre(2d+1)#,sample_points_chebyshev(4d,0,5)) 
    basis, samples = approximatefekete(basis, samples)

    #constraint 1: sum_k a_k x^k = <SOS21_ij, bb^T> + <SOS22_ij, xbb^T>
    PSD_part = Dict()
    free_part = Dict()
    for k=1:2d+1
        free_part[k] = -x^k
    end
    PSD_part[:SOS21] = LowRankMatPol([1],[basis[1:d+1]])
    PSD_part[:SOS22] = LowRankMatPol([x],[basis[1:d+1]])

    con1 = Constraint(1,PSD_part,free_part,samples) #y_0 = 1 = constant

    #constraint 2: SOS + (x - (r_i + r_j)^2)*SOS + sum_k a_ijk k! pi^-k L_k^{n/2-1}(pi x) = 0
    basis = [max_coef[i]^(-1) * q[i] for i = 1:length(q)] 
    samples = [x + r^2 for x in sample_points_rescaled_laguerre(2d+1)]#,sample_points_chebyshev(4d,r^2,5+r^2))
    basis, samples = approximatefekete(basis, samples)

    PSD_part = Dict()
    free_part = Dict()
    for k=1:2d+1
        free_part[k] = factorial(big(k))/BigFloat(pi)^k * laguerre(k, BigFloat(n) / 2 - 1, BigFloat(pi)*x)
    end

    #for three blocks:
    PSD_part[:SOS31] = hcat([basis[1]^2])
    #for 4 blocks:
    # PSD_part[:SOS31] = LowRankMatPol([R(1)],[basis[1:d+1]])
    PSD_part[:SOS32] = LowRankMatPol([x-r^2],[basis[1:d+1]])
    constant = -laguerre(0, BigFloat(n) / 2 - 1, BigFloat(pi)*x)
    con2 = Constraint(constant,PSD_part,free_part,samples)

    #objective: M
    freedict = Dict()
    for k=1:2d+1
        freedict[k] = spherevolume(n, BigFloat(r) / 2)*factorial(big(k))/BigFloat(pi)^k * laguerre(k, BigFloat(n) / 2 - 1, BigFloat(0))
    end
    constant = spherevolume(n, BigFloat(r) / 2)*laguerre(0, BigFloat(n) / 2 - 1, BigFloat(0))
    obj = Objective(constant, Dict(), freedict)
    
    #NOTE: these numbers become extremely large for large k. So no wonder that solvers have issues with that

    problem = Problem(Minimize(obj), [con1,con2])
    status, primalsol, dualsol, time, errorcode = solvesdp(problem; prec=prec, kwargs...)
    return problem, primalsol, dualsol
end

end # of module