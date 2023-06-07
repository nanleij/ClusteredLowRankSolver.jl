# We can as well work with X[j][l] (i.e. X::Vector{Vector{ArbMatrix}}}). This is only convenient for defining dot(X,Y)
# We don't use any methods of BlockDiagonal, so for speed it doesn't matter


struct SolverFailure <: Exception
    msg::String
end

@doc raw"""
	solvesdp(sdp; kwargs...)

Solve the clustered SDP with low-rank constraint matrices.

Solve the following sdp:

```math
\begin{aligned}
  \max & ∑_{j=1}^J ⟨ C^j, Y^j⟩  + ⟨ b, y ⟩ && \\
  &⟨ A^{j}_*, Y^j⟩ + B^j y = c^j,	 && j=1,\ldots,J \\
  &Y^j ⪰ 0,&& j=1,…,J,
\end{aligned}
```
where we optimize over the free variables ``y`` and the PSD block matrices
``Y^j = diag(Y^{j,1}, ..., Y^{j,L_j})``, and ``⟨A_*^j, Y^j⟩`` denotes the vector with entries ``⟨A_p^j, Y^j⟩``.
The matrices ``A^j_p`` have the same block structure as ``Y^j``. Every ``A^{j,l}`` can have several equal-sized blocks ``A^{j,l}_{r,s}``.
The smallest blocks have a low rank structure.

Keyword arguments:
  - `prec` (default: `precision(BigFloat)`): the precision used
  - `gamma` (default: `0.9`): the step length reduction; a maximum step length of α reduces to a step length of `max(gamma*α,1)`
  - `beta_(in)feasible` (default: `0.1` (`0.3`)): the amount mu is tried to be reduced by in each iteration, for (in)feasible solutions
  - `omega_p/d` (default: `10^10`): the starting matrix variable for the primal/dual is `omega_p/d*I`
  - `maxiterations` (default: `500`): the maximum number of iterations
  - `duality_gap_threshold` (default: `10^-15`): how near to optimal the solution needs to be
  - `primal/dual_error_threshold` (default:`10^-30`): how feasible the primal/dual solution needs to be
  - `max_complementary_gap` (default: `10^100`): the maximum of <X,Y>/#rows(X) allowed
  - `need_primal_feasible/need_dual_feasible` (default: `false`): terminate when the solution is primal/dual feasible
  - `verbose` (default: `true`): print information after every iteration if true
  - `step_length_threshold` (default: `10^-7`): the minimum step length allowed
  - `initial_solutions` (default: `[]`): if x,X,y,Y are given, use that instead of omega_p/d * I for the initial solutions
"""
function solvesdp(
    sdp::ClusteredLowRankSDP,
    threadinginfo::ThreadingInfo = ThreadingInfo(sdp); # the order of j and (j,l) used for threading
    prec = precision(BigFloat), # The precision used in the algorithm.
    maxiterations = 500,
    beta_infeasible = Arb(3,prec=prec) / 10, # try to decrease mu by this factor when infeasible
    beta_feasible = Arb(1,prec=prec) / 10, # try to decrease mu by this factor when feasible
    gamma = Arb(9,prec=prec) / 10, # this fraction of the maximum possible step size is used
    omega_p = Arb(10,prec=prec)^(10), # initial size of the primal PSD variables
    omega_d = Arb(10,prec=prec)^(10), # initial size of the dual PSD variables
    duality_gap_threshold = Arb(10,prec=prec)^(-15), # how near to optimal does the solution need to be
    primal_error_threshold = Arb(10,prec=prec)^(-30),  # how feasible does the primal solution need to be
    dual_error_threshold = Arb(10,prec=prec)^(-30), # how feasible does the dual solution need to be
    max_complementary_gap = Arb(10,prec=prec)^100, # the maximum of <X,Y>/#rows(X)
    need_primal_feasible = false, # terminate when the solution is primal feasible
    need_dual_feasible = false, # terminate when the solution is dual feasible
    verbose = true, # false: print nothing, true: print information after each iteration
    step_length_threshold = Arb(10,prec=prec)^(-7), # quit if the one of the step lengths is shorter than this, indicating precision errors or infeasibility
    initial_solutions = [], # initial solutions of the right format, in the order x,X,y,Y. This can give errors if the sizes are incorrect or when the matrices are not PSD
    #experimental & testing:
    matmul_prec = prec, # precision for matrix multiplications for the bilinear pairings. A lower precision increases speed and probably decreases memory consumption, but also increases the minimum errors
	testing = false, # print the times of the first two iterations. This is for testing purposes
)
    # the default values mostly come from Simmons-Duffin original paper, or from the default values of SDPA-GMP (slow but stable mode)

	#NOTE: Because we use Arb through Arblib.jl, the code might look a bit like C instead of Julia.
	# There is an issue on the Arblib.jl GitHub about using the MutableArithmetic api to use e.g.
	#	MA.@rewrite a + b + c
	# instead of
	#	Arblib.add!(a,a,b)
	#	Arblib.add!(a,a,c)
	# which might make it easier to read.


    # convert to Arbs & the required precision:
    omega_p,
    omega_d,
    gamma,
    beta_feasible,
    beta_infeasible,
    duality_gap_threshold, #we don't really need the thresholds to be arbs?
    primal_error_threshold,
    dual_error_threshold = (
        Arb.(
            [
                omega_p,
                omega_d,
                gamma,
                beta_feasible,
                beta_infeasible,
                duality_gap_threshold,
                primal_error_threshold,
                dual_error_threshold,
            ],
            prec = prec,
        )
    )
    sdp = convert_to_prec(sdp,prec) # convert the sdp to the precision we use in the solver.

    # The algorithm:
    # initialize:
        # 1): choose initial point q = (0, Ω_p*I, 0, Ω_d*I) = (x,X,y,Y), with Ω>0
    # main loop:
        # 2): compute residues  P = ∑_i A_i x_i - X - C, p = b -B^Tx, d = c- <A_* Y> - By and R = μI - XY
        # 3): Take μ = <X,Y>/K, and μ_p = beta_p μ with beta_p = 0 if q is primal & dual feasible, beta_infeasible otherwise
        # 4): solve the system for the search direction (dx,dX, dy,dY), with R = μ_p I - XY
        # 5): compute corrector deformation μ_c = beta_c μ:
            # r = <(X+dX),(Y+dY)>/(μK)
            # beta = r^2 if r<1, r otherwise
            # beta_c = min( max( beta_feasible, beta),1) if primal & dual feasible, max(beta_infeasible,beta) otherwise
        # 6): solve the system for the search direction (dx,dX, dy,dY), with R = μ_c I - XY
        # 7): compute step lengths: α_p = min(γ α(X,dX),1), α_d = min(γ α(Y,dY),1)
            # with α(M,dM) =  max(0, -eigmin(M)/eigmin(dM)) ( = 0 if eigmin(dM)> 0).
        # 8): do the steps: x,X -> x,X + α_p dx,dX and y,Y -> y,Y + α_d dy,dY
    # Repeat from step 2

    # get the blocksizes to initiate the variables
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

    # step 1: initialize. We may pass a solution from a previous run
    if length(initial_solutions) != 4 # we need the whole solution, x,X,y,Y, otherwise we initialize the default
        x = ArbRefMatrix(sum(size.(sdp.c,1)), 1, prec = prec) # all tuples (j,r,s,k), equals size of S.
        X = BlockDiagonal([
            BlockDiagonal([
                ArbRefMatrix(
                        subblocksizes[j][l]*size(sdp.A[j][l],1),
                        subblocksizes[j][l]*size(sdp.A[j][l],1),
                    prec = prec,
                ) for l in eachindex(sdp.A[j])
            ]) for j in eachindex(sdp.A)
        ])
        y = ArbRefMatrix(size(sdp.b,1), 1, prec = prec)
        Y = BlockDiagonal([
            BlockDiagonal([
                ArbRefMatrix(
                        subblocksizes[j][l]*size(sdp.A[j][l],1),
                        subblocksizes[j][l]*size(sdp.A[j][l],1),
                    prec = prec,
                ) for l in eachindex(sdp.A[j])
            ]) for j in eachindex(sdp.A)
        ])
        #set the X,Y matrices to omega_p resp. omega_d * I
        for j in eachindex(sdp.A)
            for l in eachindex(sdp.A[j])
                Arblib.one!(X.blocks[j].blocks[l])
                Arblib.mul!(X.blocks[j].blocks[l],X.blocks[j].blocks[l], omega_p)
                Arblib.one!(Y.blocks[j].blocks[l])
                Arblib.mul!(Y.blocks[j].blocks[l],Y.blocks[j].blocks[l], omega_d)
            end
        end
    else
        #initial solutions = [x,X,y,Y].
        x = ArbRefMatrix(initial_solutions[1],prec=prec)
        Arblib.get_mid!(x,x)
        y = ArbRefMatrix(initial_solutions[3],prec=prec)
        Arblib.get_mid!(y,y)

        X = BlockDiagonal([BlockDiagonal([ArbRefMatrix(initial_solutions[2].blocks[j].blocks[l], prec=prec) for l in eachindex(sdp.A[j])]) for j in eachindex(sdp.A)])
        Y = BlockDiagonal([BlockDiagonal([ArbRefMatrix(initial_solutions[4].blocks[j].blocks[l], prec=prec) for l in eachindex(sdp.A[j])]) for j in eachindex(sdp.A)])

        for (j,l) in threadinginfo.jl_order
            Arblib.get_mid!(X.blocks[j].blocks[l],X.blocks[j].blocks[l])
            Arblib.get_mid!(Y.blocks[j].blocks[l],Y.blocks[j].blocks[l])
        end
        #check sizes:
        @assert size(x) == (sum(size.(sdp.c,1)),1)
        @assert size(y) == (size(sdp.b,1),1)
        @assert all([size.(X.blocks[j].blocks,1) == subblocksizes[j] .* size.(sdp.A[j],1) for j in eachindex(sdp.A)])
        @assert all([size.(Y.blocks[j].blocks,1) == subblocksizes[j] .* size.(sdp.A[j],1) for j in eachindex(sdp.A)])
        #we don't check for PSDness, but that's implicitely done in the algorithm
    end

    #separate the high-rank blocks from the low-rank ones.
    # Precompute the matrices for the bilinear pairings and the matrices used for ∑_i x_i A_i and <A^j_*, Y>
    leftvecs_pairings, rightvecs_pairings, pointers_left, pointers_right,high_ranks = precompute_matrices_bilinear_pairings(sdp,subblocksizes)
    #vecs_left and vecs_right are actually just parts of leftvecs_pairings and rightvecs_pairings ? 
    # note that we can take a flag 'symmetric' if vecsleft == vecsright for all j,l,r,s; then we only have to make the matrix for half of them
    # Note that we can forget about either the leftvecs_pairings or vecs_left, and possibly use the pointers (if we forget about vecs_left)
    # in either case, we save half the memory we use here.
    vecs_left = [[[ArbRefMatrix(0,0) for r=1:size(sdp.A[j][l],1), s=1:size(sdp.A[j][l],2)] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]
    vecs_right = [[[ArbRefMatrix(0,0) for r=1:size(sdp.A[j][l],1), s=1:size(sdp.A[j][l],2)] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]
    max_s = maximum(size(sdp.A[j][l],2) for j in eachindex(sdp.A) for l in eachindex(sdp.A[j]))
    maxsize = [[maximum(size(leftvecs_pairings[j][l][s],1) for j in eachindex(sdp.A) for l in eachindex(sdp.A[j]) for s=1:size(sdp.A[j][l],2) if s == sf),
                maximum(size(rightvecs_pairings[j][l][s],2) for j in eachindex(sdp.A) for l in eachindex(sdp.A[j]) for s=1:size(sdp.A[j][l],2) if s == sf)] for sf=1:max_s]
    max_blocksize = maximum(size(Y.blocks[j].blocks[l],2) for j in eachindex(sdp.A) for l in eachindex(sdp.A[j]))
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            if high_ranks[j][l]
               continue
            end 
            delta = div(size(Y.blocks[j].blocks[l],1),size(sdp.A[j][l],1))
            for r=1:size(sdp.A[j][l],1), s=1:size(sdp.A[j][l],2)
                cur_right = hcat([sdp.A[j][l][r,s][p].rightevs[i] for p in keys(sdp.A[j][l][r,s]) for i=1:length(sdp.A[j][l][r,s][p].rightevs)]...)
                cur_left = hcat([sdp.A[j][l][r,s][p].leftevs[i] for p in keys(sdp.A[j][l][r,s]) for i=1:length(sdp.A[j][l][r,s][p].leftevs)]...)
                if size(cur_left,1) == 0
                    vecs_left[j][l][r,s] = ArbRefMatrix(delta,0,prec=prec)
                else
                    vecs_left[j][l][r,s] = cur_left
                end
                if size(cur_right,1) == 0
                    vecs_right[j][l][r,s] = ArbRefMatrix(delta,0,prec=prec)
                else
                    vecs_right[j][l][r,s] = cur_right
                end
            end
        end
    end

    #step 2
    #loop initialization: compute or set the initial values, and print the header
    iter = 1
    if verbose
		@printf("%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n",
            "iter","time(s)","μ","P-obj","D-obj","gap","P-error","p-error","d-error","α_p","α_d","beta"
        )
	end
    alpha_p = alpha_d = Arb(0, prec = prec)
    mu = dot(X, Y) / size(X, 1)
    # Preallocation. In principle we can collect them in a struct or something like that and use them that way.
	# Then we only need to give e.g. Variables and Preallocated to functions instead of e.g. P,p,d, dX,dY,dx,dy
    # We overwrite R, X_inv, S and A_Y in each iteration.
    R = similar(X)
    X_inv = similar(X)
    S = [ArbRefMatrix(size(sdp.c[j],1), size(sdp.c[j],1), prec = prec) for j in eachindex(sdp.c)]
    A_Y = [[
        [ArbRefMatrix(size(vecs_left[j][l][r,s],2), 1, prec = prec) for r=1:size(sdp.A[j][l],1), s=1:size(sdp.A[j][l],2)] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)
    ]
    part_r = ArbRefMatrix(max_blocksize,maximum(maxsize[r][2] for r=1:max_s),prec=matmul_prec)
    P = similar(X)
    d = similar(x)
    p = similar(y)
    dX = similar(X)
    dY = similar(X)
    dx = similar(x)
    dy = similar(y)
    bilinear_pairings_Y = [ArbRefMatrix(maxsize[r][1], maxsize[s][2], prec=matmul_prec) for r=1:max_s, s = 1:max_s]
    bilinear_pairings_Xinv = [ArbRefMatrix(maxsize[r][1], maxsize[s][2], prec=matmul_prec) for r=1:max_s, s = 1:max_s]

    LinvB = [
        ArbRefMatrix(size(sdp.c[j],1), size(sdp.b,1), prec = prec) for j in eachindex(sdp.A)
    ]
    Q = ArbRefMatrix(size(sdp.b,1),size(sdp.b,1),prec=prec)
    tempX = [similar(X),similar(X)] # we need this scratch space several times each iteration
    #errors and feasibility
    p_obj = compute_primal_objective(sdp, x)
    d_obj = compute_dual_objective(sdp, y, Y)
    dual_gap = compute_duality_gap(sdp, x, y, Y)
    time_res = @elapsed begin
        #goes wrong here (probably due to improper window or something like that)
        compute_residuals!(sdp, x, X, y, Y, P, p, d, threadinginfo,vecs_left,vecs_right,high_ranks)
    end
    primal_error = compute_primal_error(P, p)
    dual_error = compute_dual_error(d)
    pd_feas = check_pd_feasibility(
        primal_error,
        dual_error,
        primal_error_threshold,
        dual_error_threshold,
    )
    error_code = [0] #success
    #we keep track of allocations and time of some of the parts of the algorithm
    allocs = zeros(6)
    timings = zeros(17)
    time_start = time()
    try #catch InterruptExceptions & SolverFailures (so that we still return something)
    @time while (
        !terminate(
            dual_gap,
            primal_error,
            dual_error,
            duality_gap_threshold,
            primal_error_threshold,
            dual_error_threshold,
            need_primal_feasible,
            need_dual_feasible,
        )
    )
        if iter > maxiterations
            println("The maximum number of iterations has been reached.")
            error_code[1] = 2
            break
        end
        GC.gc(false) #not sure if needed
        #step 3
        mu = dot(X, Y) / size(X, 1)
        mu_p = pd_feas ? zero(mu) : beta_infeasible * mu # zero(mu) keeps the precision
        if mu > max_complementary_gap
            println("The maximum complementary gap has been exceeded (mu = $mu).")
            error_code[1] = 3
            break
        end
        #step 4
        time_R = @elapsed begin
            compute_residual_R!(R, X, Y, mu_p, tempX, threadinginfo)
        end

        # We compute the cholesky decomposition, and then solve a triangular system when we use it.
        time_inv = @elapsed begin
            Threads.@threads for (j,l) in threadinginfo.jl_order
                status = approx_cholesky!(X_inv.blocks[j].blocks[l], X.blocks[j].blocks[l],)
                Arblib.get_mid!(
                    X_inv.blocks[j].blocks[l],
                    X_inv.blocks[j].blocks[l],
                ) #ignore the error bounds. Most of the error bounds will be 0 due to the use of approx_cholesky already
                if status == 0
                    @show j,l
                    throw(SolverFailure(
                        "The cholesky decomposition of X was not computed correctly. Try again with higher precision",
                    ))
                end
            end
        end

        # Compute the decomposition which is used to solve the system of equations for the search directions.
		# We also keep A_Y, which is used for <A_*, Y>
        allocs[1] += @allocated begin
            time_decomp = @elapsed begin
                time_schur, time_cholS, time_LinvB, time_Q, time_cholQ =
                    compute_T_decomposition!(sdp, S,A_Y,X_inv, Y,bilinear_pairings_Y,bilinear_pairings_Xinv, LinvB,Q, tempX, threadinginfo,leftvecs_pairings,rightvecs_pairings,pointers_left,pointers_right,high_ranks, part_r, prec=matmul_prec)
            end
        end

        # Compute the residuals
        allocs[2] += @allocated begin
            time_res = @elapsed begin
                compute_residuals!(sdp, x, X, y, (Y, A_Y), P, p, d, threadinginfo,vecs_left,vecs_right,high_ranks)
            end
        end

        # Compute the predictor search direction
        allocs[3] += @allocated begin
            time_predictor_dir = @elapsed begin
                times_predictor_in =
                    compute_search_direction!(sdp,dx, dX, dy, dY,  P, p, d, R, X_inv, Y, tempX, threadinginfo, S, LinvB,Q, vecs_left,vecs_right,high_ranks)
            end
        end

        # step 5: compute mu_c
        #TODO: use threads
        r = (dot(X, Y) + dot(X, dY) + dot(dX, Y) + dot(dX, dY)) / (mu * size(X, 1)) 
        beta = r < 1 ? r^2 : r
        beta_c =
            pd_feas ? min(max(beta_feasible, beta), Arb(1, prec = prec)) :
            max(beta_infeasible, beta)
        mu_c = beta_c * mu

        # step 6: the corrector search direction
		# Compute R_c
        time_R += @elapsed begin
            compute_residual_R!(R, X, Y, mu_c, dX, dY, tempX, threadinginfo)
        end
		primal_error = compute_primal_error(P, p)
		dual_error = compute_dual_error(d)
		pd_feas = check_pd_feasibility(
            primal_error,
            dual_error,
            primal_error_threshold,
            dual_error_threshold)


        # Compute the corrector search direction
        allocs[4] += @allocated begin
            time_corrector_dir = @elapsed begin
                times_corrector_in =
                    compute_search_direction!(sdp,dx, dX, dy, dY,  P, p, d, R, X_inv, Y, tempX, threadinginfo, S, LinvB,Q, vecs_left,vecs_right,high_ranks)
            end
        end


        # step 7: compute the step lengths
        allocs[5] += @allocated begin
            time_alpha = @elapsed begin
                alpha_p = compute_step_length(X, dX, gamma, tempX, threadinginfo)
                alpha_d = compute_step_length(Y, dY, gamma, tempX, threadinginfo)
            end
        end



        # if the precision is not high enough, or if there are other problems, the step lengths might be extremely low (e.g. 10^-8 or lower)
        if min(BigFloat(alpha_d),BigFloat(alpha_p)) < step_length_threshold
            min_step = min(Float64(alpha_d),Float64(alpha_p))
            println("The step length (", min_step, ") was too short, possible reasons include precision issues and infeasible problems.")
            println("Another reason might be that the current solution is in a difficult area. In that case you can try decreasing the parameter `step_length_threshold` and/or `gamma`.")
            error_code[1] = 4
            break
        else
            # The steplengths are long enough, so we update the variables.
            # if the current solution is primal ánd dual feasible, we follow the search direction exactly.
            # (this follows the code for SDPB)
            if pd_feas
                alpha_p = min(alpha_p, alpha_d)
                alpha_d = alpha_p
            end
            #step 8: perform the step
            Arblib.addmul!(x, dx, alpha_p)
            Arblib.addmul!(y, dy, alpha_d)
            Arblib.get_mid!(x,x)
            Arblib.get_mid!(y,y)
            Threads.@threads for (j,l) in threadinginfo.jl_order
                Arblib.addmul!(X.blocks[j].blocks[l], dX.blocks[j].blocks[l], alpha_p)
                Arblib.get_mid!(X.blocks[j].blocks[l], X.blocks[j].blocks[l])

                Arblib.addmul!(Y.blocks[j].blocks[l], dY.blocks[j].blocks[l], alpha_d)
                Arblib.get_mid!(Y.blocks[j].blocks[l], Y.blocks[j].blocks[l])
            end
        end

        # We save the times of everything except for the first iteration, as they may include compile time
        if iter >= 2
            timings[1] += time_decomp
            timings[2] += time_predictor_dir
            timings[3] += time_corrector_dir
            timings[4] += time_alpha
            timings[5] += time_inv
            timings[6] += time_R
            timings[7] += time_res
            timings[8:12] .+= [time_schur, time_cholS, time_LinvB, time_Q, time_cholQ]
            timings[13:17] .+= times_predictor_in .+ times_corrector_in
        elseif testing && verbose #if testing, the times of the first few iterations may be interesting
            println(
                "decomp:",
                time_decomp,
                ". directions:",
                time_predictor_dir + time_corrector_dir,
                ". steplength:",
                time_alpha,
            )
            println(
                "schur:",
                time_schur,
                " cholS:",
                time_cholS,
                " LinvB:",
                time_LinvB,
                " Q:",
                time_Q,
                " cholQ:",
                time_cholQ,
            )
            println("X inv:", time_inv, ". R:", time_R, ". residuals p,P,d:", time_res)
        end
        # print the objectives of the start of the iteration, imitating simmons duffin
        # This might be a bit weird, because we only know the step lengths at the end of the iteration
        if verbose
			@printf(
                "%5d %8.1f %11.3e %11.3e %11.3e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",
                iter,
                time() - time_start,
                BigFloat(mu),
                BigFloat(p_obj),
                BigFloat(d_obj),
                BigFloat(dual_gap),
                BigFloat(compute_error(P)),
                BigFloat(compute_error(p)),
                BigFloat(compute_error(d)),
                BigFloat(alpha_p),
                BigFloat(alpha_d),
                beta_c
            )
        end

        # Compute the new objectives, for the new iteration
        allocs[6]+= @allocated begin
            p_obj = compute_primal_objective(sdp, x)
            d_obj = compute_dual_objective(sdp,y, Y)
            dual_gap = compute_duality_gap(p_obj, d_obj)
        end #of allocs


        iter += 1
    end
    catch e
        println("A(n) $(typeof(e)) occurred.")
        if hasfield(typeof(e),:msg)
            println(e.msg)
        end
        println("We return the current solution and optimality status.")
        error_code[1] = 1 #general errors
    end #of try/catch
    time_total = time() - time_start #this may include compile time
	results = CLRSResults(x, X, y, Y, compute_primal_objective(sdp, x), compute_dual_objective(sdp,y, Y), sdp.matrix_coeff_names, sdp.free_coeff_names)

    if verbose
		@printf(
            "%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n",
            "iter",
            "time(s)",
            "μ",
            "P-obj",
            "D-obj",
            "gap",
            "P-error",
            "p-error",
            "d-error",
            "α_p",
            "α_d",
            "beta"
        )
		if testing
	        #print the total time needed for every part of the algorithm
	        println(
	            "\nTime spent: (The total time may include compile time. The first few iterations are not included in the rest of the times)",
	        )
	        @printf(
	            "%11s %11s %11s %11s %11s %11s %11s %11s\n",
	            "total",
	            "Decomp",
	            "predict_dir",
	            "correct_dir",
	            "alpha",
	            "Xinv",
	            "R",
	            "residuals"
	        )
	        @printf(
	            "%11.5e %11.5e %11.5e %11.5e %11.5e %11.5e %11.5e %11.5e\n\n",
	            time_total,
	            timings[1:7]...
	        )
	        println("Time inside decomp:")
	        @printf(
	            "%11s %11s %11s %11s %11s\n",
	            "schur",
	            "chol_S",
	            "comp LinvB",
	            "comp Q",
	            "chol_Q"
	        )
	        @printf("%11.5e %11.5e %11.5e %11.5e %11.5e\n\n", timings[8:12]...)

	        println("Time inside search directions (both predictor & corrector step)")
	        @printf(
	            "%11s %11s %11s %11s %11s\n",
	            "calc Z",
	            "calc rhs x",
	            "solve system",
	            "calc dX",
	            "calc dY"
	        )
	        @printf("%11.5e %11.5e %11.5e %11.5e %11.5e\n\n", timings[13:17]...)

            println("Allocations in several functions:")
            @printf(
                "%11s %11s %11s %11s %11s %11s\n",
                "Decomp",
                "Residuals",
                "Predictor",
                "Corrector",
                "Step length",
                "Errors",
            )
            @printf("%11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n\n", allocs...)
        end
		println("\nPrimal objective:", results.primal_objective)
		println("Dual objective:", results.dual_objective)
		println("Duality gap:", compute_duality_gap(results.primal_objective, results.dual_objective))

    end


    # Get the status from the results
    if pd_feas && dual_gap < duality_gap_threshold
        # optimal
        status = Optimal()
    elseif (pd_feas && dual_gap < 10^-8) || (primal_error < 10^-15 && dual_error < 10^-15 && dual_gap < 10^-8)
        # How do we define near optimal? Maybe we should remove this
        status = NearOptimal()
    elseif pd_feas
        status = Feasible()
    elseif dual_error < dual_error_threshold && !pd_feas
        status = DualFeasible()
    elseif primal_error < primal_error_threshold && !pd_feas
        status = PrimalFeasible()
    else
        status = NotConverged()
    end

    return status, results, time_total, error_code[1] #maybe wrap time & error code in something like SolverStatistics?
end

"""Compute the primal objective <c, x> + constant"""
function compute_primal_objective(sdp, x)
    if sdp.maximize
        return dot_c(sdp, x) + sdp.constant
    else
        return -dot_c(sdp, x) + sdp.constant
    end
end

"""Compute the dual objective <C,Y> + <b,y> + constant"""
function compute_dual_objective(sdp,y, Y)
    return dot(sdp.C, Y) + dot(sdp.b, y) + sdp.constant
end

"""Compute the error (max abs (P_ij)) of a blockdiagonal matrix"""
function compute_error(P::BlockDiagonal)
    max_P = Arb(0, prec = precision(P))
    for b in P.blocks
        Arblib.max!(max_P, compute_error(b), max_P)
    end
    return max_P
end

"""Compute the error (max abs(d_ij)) of an ArbMatrix"""
function compute_error(d::ArbRefMatrix)
    max_d = Arb(0, prec = precision(d))
    abs_temp = Arb(0, prec = precision(d))
    for i = 1:size(d, 1)
        for j = 1:size(d, 2)
            Arblib.max!(max_d, max_d, Arblib.abs!(abs_temp,d[i,j]))
        end
    end
    return max_d
end

"""Compute the primal error"""
function compute_primal_error(P, p)
    max_p = compute_error(p)
    max_P = compute_error(P)
    return Arblib.max!(max_p, max_p, max_P)
end

"""Compute the dual error"""
compute_dual_error(d) = compute_error(d)

"""Compute the duality gap (primal_obj - dual_obj)/max{1, |primal_obj+dual_obj|}"""
function compute_duality_gap(sdp, x, y, Y)
    primal_objective = compute_primal_objective(sdp,x)
    dual_objective = compute_dual_objective(sdp,y,Y)
    return compute_duality_gap(primal_objective, dual_objective)
end

function compute_duality_gap(primal_objective, dual_objective)
    return abs(primal_objective - dual_objective) /
           max(one(primal_objective), abs(primal_objective + dual_objective))
end

"""Compute <c,x> where c is distributed over constraints"""
function dot_c(sdp, x)
    res = Arb(0, prec = precision(x))
    x_idx = 1
    for j = 1:length(sdp.c)
        for i = 1:length(sdp.c[j])
            Arblib.addmul!(res, sdp.c[j][i,1], x[x_idx,1])
            x_idx += 1
        end
    end
    return res
end

"""Compute the dual residue d = c - <A_*, Y> - By"""
function calculate_res_d!(sdp,y,Y,d,leftvecs_pairings,rightvecs_pairings,high_ranks)
    cur_idx = 0
    w = ArbRefMatrix(0,0)
    for j in eachindex(sdp.c)
        Arblib.window_init!(w, d, cur_idx, 0,cur_idx+size(sdp.c[j],1), 1)
        
        Arblib.approx_mul!(w, sdp.B[j], y)
        Arblib.neg!(w,w)
        Arblib.add!(w,w,sdp.c[j])
        Arblib.window_clear!(w)
        # d[cur_idx+1:cur_idx+size(sdp.c[j],1),1] = sdp.c[j]
        cur_idx += size(sdp.c[j],1)
    end

    Arblib.sub!(d,d,trace_A(sdp,Y,leftvecs_pairings,rightvecs_pairings,high_ranks))
    return d
end

"""Compute the residuals P,p and d."""
function compute_residuals!(sdp, x, X, y, Y, P, p, d,threadinginfo,leftvecs_pairings,rightvecs_pairings,high_ranks)
    # P = ∑_i x_i A_i - X - C, (+ C if we are minimizing)
    compute_weighted_A!(P, sdp, x,leftvecs_pairings,high_ranks)
    Threads.@threads for (j,l) in threadinginfo.jl_order
        Arblib.sub!(P.blocks[j].blocks[l],P.blocks[j].blocks[l],X.blocks[j].blocks[l])
        if sdp.maximize  # normal
            Arblib.sub!(P.blocks[j].blocks[l],P.blocks[j].blocks[l], sdp.C.blocks[j].blocks[l])
        else # we use -C in the program when minimizing
            Arblib.add!(P.blocks[j].blocks[l],P.blocks[j].blocks[l], sdp.C.blocks[j].blocks[l])
        end
        Arblib.get_mid!(P.blocks[j].blocks[l], P.blocks[j].blocks[l])
    end

    # d = c - <A_*, Y> - By
    calculate_res_d!(sdp,y,Y,d,leftvecs_pairings,rightvecs_pairings,high_ranks)
    Arblib.get_mid!(d, d)

    # p = b - B^T x (-b if we are minimizing)
    Arblib.zero!(p)
    # We do it per thread separately, but they need to be added together which cannot directly be done with threading
    p_added = [similar(p) for j in eachindex(sdp.B)]
    Threads.@threads for j in threadinginfo.j_order
        j_idx = sum(size.(sdp.c[1:j-1],1))
        approx_mul_transpose!(p_added[j],sdp.B[j], x[j_idx+1:j_idx+size(sdp.c[j],1),:])
    end
    for j in eachindex(p_added)
        Arblib.sub!(p,p,p_added[j])
    end
    if sdp.maximize #normal
        Arblib.add!(p,p,sdp.b)
    else
        # we use -b in the program when minimizing
        Arblib.sub!(p,p,sdp.b)
    end
    Arblib.get_mid!(p, p)
    return nothing # we modify P,p and d
end

"""Determine whether the main loop should terminate or not"""
function terminate(
    duality_gap,
    primal_error,
    dual_error,
    duality_gap_threshold,
    primal_error_threshold,
    dual_error_threshold,
    need_primal_feasible,
    need_dual_feasible,
)
    #NOTE: We might also need termination criteria for when the problem is infeasible. e.g. a too large primal/dual objective
    # We convert to BigFloats to avoid unexpected results due to large error bounds.
    # This is (probably) only important when increasing the precision during the solving, or not at all
    duality_gap_opt = BigFloat(duality_gap) < BigFloat(duality_gap_threshold)
    primal_feas = BigFloat(primal_error) < BigFloat(primal_error_threshold)
    dual_feas = BigFloat(dual_error) < BigFloat(dual_error_threshold)
    if need_primal_feasible && primal_feas
        println("Primal feasible solution found")
        return true
    end
    if need_dual_feasible && dual_feas
        println("Dual feasible solution found")
        return true
    end
    if primal_feas && dual_feas && duality_gap_opt
        println("Optimal solution found")
        return true
    end
    return false
end

"""Check primal and dual feasibility"""
function check_pd_feasibility(primal_error, dual_error, primal_error_threshold, dual_error_threshold)
    primal_feas = primal_error < primal_error_threshold
    dual_feas = dual_error < dual_error_threshold
    return primal_feas && dual_feas
end


"""Compute the residual R, with or without second order term """
function compute_residual_R!(R, X, Y, mu,tempX, threadinginfo)
    # R = mu*I - X * Y
    Threads.@threads for (j,l) in threadinginfo.jl_order
        Arblib.one!(R.blocks[j].blocks[l])
        Arblib.mul!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], mu)
        Arblib.approx_mul!(tempX[1].blocks[j].blocks[l], X.blocks[j].blocks[l], Y.blocks[j].blocks[l])
        Arblib.sub!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], tempX[1].blocks[j].blocks[l])
    end
    return R
end

function compute_residual_R!(R, X, Y, mu, dX, dY, tempX, threadinginfo)
    # R = mu*I - X Y - dX dY
    Threads.@threads for (j,l) in threadinginfo.jl_order
        Arblib.one!(R.blocks[j].blocks[l])
        Arblib.mul!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], mu)
        Arblib.approx_mul!(tempX[1].blocks[j].blocks[l], X.blocks[j].blocks[l], Y.blocks[j].blocks[l])
        Arblib.sub!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], tempX[1].blocks[j].blocks[l])
        Arblib.approx_mul!(tempX[1].blocks[j].blocks[l], dX.blocks[j].blocks[l], dY.blocks[j].blocks[l])
        Arblib.sub!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], tempX[1].blocks[j].blocks[l])
    end
    return R
end

function precompute_matrices_bilinear_pairings(sdp, subblocksizes; prec = precision(sdp.b))
    #In this function we precompute the matrices [v^j,l_p,r,s ...] for the matrix multiplications Vl*Y*Vr and Vl*X^-1 * Vr, and the indexing j,l,p,r,s -> column/row
    # This only has to be done once, so performance does not have a high priority here.
    # We try to remove duplicates, because that is the main difference between the old and new format for things like binary sphere packing.
    leftvecs = [[ArbRefMatrix[] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]
    rightvecs = [[ArbRefMatrix[] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]
    high_ranks = [Bool[false for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]

    w = ArbRefMatrix(0,0,prec=prec)
    # Indexing: j,l,r
    # For every (s,p,rank) combination for this j,l,r we have an entry pointing towards the vector from leftvecs resp rightvecs.
    pointers_right = [[[Dict{Tuple{Int,Int,Int},Int}() for r=1:size(sdp.A[j][l],1)] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]
    pointers_left = [[[Dict{Tuple{Int,Int,Int},Int}() for r=1:size(sdp.A[j][l],1)] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            high_ranks[j][l] = any(typeof(sdp.A[j][l][i][p]) != LowRankMat for i in eachindex(sdp.A[j][l]) for p in keys(sdp.A[j][l][i]))
            for r=1:size(sdp.A[j][l],1)
                if high_ranks[j][l] #for high rank matrices we don't have left/right vecs
                    push!(rightvecs[j][l], ArbRefMatrix(subblocksizes[j][l],0,prec=prec))
                    push!(leftvecs[j][l], ArbRefMatrix(subblocksizes[j][l],0,prec=prec))
                    pointers_right[j][l][r] = Dict{Tuple{Int,Int,Int},Int}()
                    pointers_left[j][l][r] = Dict{Tuple{Int,Int,Int},Int}()
                    continue
                end
                # NOTE: this assumes that sdp.A[j][l][r,s][p] = transpose(sdp.A[j][l][s,r][p]). We need to make sure that that is the case.
                # This is the only place where we really need it. (and where we use these matrices)
                # vectors for this combination of j,l,r
                right = [sdp.A[j][l][r,s][p].rightevs[rnk] for s=1:size(sdp.A[j][l],2) for p in keys(sdp.A[j][l][r,s]) for rnk=1:length(sdp.A[j][l][r,s][p].eigenvalues)]
                if length(right) == 0
                    push!(rightvecs[j][l], ArbRefMatrix(subblocksizes[j][l],0,prec=prec))
                    pointers_right[j][l][r] = Dict{Tuple{Int,Int,Int},Int}() #nothing to point to, and nothing which points to anything
                else
                    #remove duplicates
                    unique_right = unique_idx(right)
                    #points to the column with the vector we want
                    pntr_right = Dict((s,p,rnk) => findfirst(x-> right[x]==right[i], unique_right) for (i,(s,p,rnk)) in enumerate([(s,p,rnk) for s=1:size(sdp.A[j][l],2) for p in keys(sdp.A[j][l][r,s]) for rnk=1:length(sdp.A[j][l][r,s][p].eigenvalues)]))
                    pointers_right[j][l][r] = pntr_right
                    unique_right_arb = ArbRefMatrix(size(right[1],1),length(unique_right), prec=prec)
                    for (i, i_vec) in enumerate(unique_right)
                        unique_right_arb[:, i] = right[i_vec]
                    end
                    # cur_rvec = ArbRefMatrix(size(unique_right_arb,1), size(unique_right_arb,2),prec=prec)
                    Arblib.set_round!.(unique_right_arb, unique_right_arb) #round to the wanted precision
                    Arblib.get_mid!(unique_right_arb, unique_right_arb)
                    push!(rightvecs[j][l], unique_right_arb)
                end

                # We do the same for the left eigenvectors
                left = [sdp.A[j][l][r,s][p].leftevs[rnk] for s=1:size(sdp.A[j][l],2) for p in keys(sdp.A[j][l][r,s]) for rnk=1:length(sdp.A[j][l][r,s][p].eigenvalues)]
                if length(left) == 0
                    push!(leftvecs[j][l], ArbRefMatrix(subblocksizes[j][l],0,prec=prec))
                    pointers_left[j][l][r] = Dict{Tuple{Int,Int,Int},Int}()
                else
                    #remove duplicates
                    unique_left = unique_idx(left)
                    #points to the column with the vector we want
                    pntr_left = Dict((s,p,rnk) => findfirst(x-> left[x]==left[i], unique_left) for (i,(s,p,rnk)) in enumerate([(s,p,rnk) for s=1:size(sdp.A[j][l],2) for p in keys(sdp.A[j][l][r,s]) for rnk=1:length(sdp.A[j][l][r,s][p].eigenvalues)]))
                    pointers_left[j][l][r] = pntr_left
                    unique_left_arb = ArbRefMatrix(length(unique_left),size(left[1],1), prec=prec)
                    for (i, i_vec) in enumerate(unique_left)
                        Arblib.window_init!(w, unique_left_arb, i-1,0,i,size(unique_left_arb,2))
                        Arblib.transpose!(w, left[i_vec])
                        Arblib.window_clear!(w)
                    end
                    Arblib.set_round!.(unique_left_arb, unique_left_arb) #round to the wanted precision
                    Arblib.get_mid!(unique_left_arb, unique_left_arb)
                    push!(leftvecs[j][l], unique_left_arb)
                end


            end
        end
    end
    return leftvecs,rightvecs,pointers_left,pointers_right, high_ranks
end

"""Compute S, integrated with the precomputing of the bilinear pairings"""
function compute_S_integrated!(S,sdp,A_Y, X_inv, Y,bilinear_pairings_Y, bilinear_pairings_Xinv, leftvecs,rightvecs,pointers_left,pointers_right,high_ranks, tempX, temppart_r;matmul_prec=precision(S[1]))
    #NOTE: somewhere in this function, the memory is allocated but not always released. Goes from 13% before to 20% just after this function.
    #   Apparently, calling 'ccall(:malloc_trim, Cvoid, (Cint,), 0)' could free (some) more memory
    prec = precision(Y)
    # We have done some preprocessing: make an ArbMatrix of the unique vectors we need to use for Y_.,r etc,
    # and pointers from p,r,s to the right column/entry. Then we can just apply the matrix multiplication, and then later pick the entry the pointer is pointing towards.

    # For <A_*, Y> we need bilinear_pairings_Y[r,s,(k,rnk),(k,rnk)]. So per r,s block, the diagonal
    if any(size(rightvecs[j][l][r],2) > 200 for j in eachindex(sdp.A) for l in eachindex(sdp.A[j]) for r=1:size(sdp.A[j][l],1))
        #We only need to do GC.gc() if there are a lot of allocations
        GC.gc()
        malloc_trim(0)
    end
    base = Sys.free_memory()/1024^3 #in GB

    temp_window = ArbRefMatrix(0,0)
    temp_window2 = ArbRefMatrix(0,0)
    part_r = ArbRefMatrix(0,0)
    for j in eachindex(sdp.A)
        Arblib.zero!(S[j])
        for l in eachindex(sdp.A[j])
            sz = size(Y.blocks[j].blocks[l],1)
            delta = div(sz, size(sdp.A[j][l],1)) #NOTE: we assume that all blocks are of the same size
            # @show sz, delta
            #Here we make a distinction between low- and high-rank blocks 
            # For high-rank blocks we do not use the sparsity (if existing) at first. This might be optimized later, using the formula F*(k) of Fujisawa et al. (1997)
            if high_ranks[j][l]
                #for high_rank matrices we don't compute A_Y. We compute the contribution to S_{pq} by
                #   dot(A[j][l][p], X^-1 * A[j][l][q] * Y)
                # high rank matrices always have one subblock
                for p in keys(sdp.A[j][l][1,1]) #Not sure if we should do this threaded or the matrix multiplications
                    # scratch = similar(X_inv.blocks[j].blocks[l])
                    Arblib.solve_cho_precomp!(tempX[1].blocks[j].blocks[l], X_inv.blocks[j].blocks[l], sdp.A[j][l][1,1][p])
                    #this goes wrong with threads because we use the same block of tempX two times 
                    matmul_threaded!(tempX[2].blocks[j].blocks[l], tempX[1].blocks[j].blocks[l], Y.blocks[j].blocks[l], prec=matmul_prec)

                    # We only do the upper triangular part here, so q >= p
                    qs = [q for q in keys(sdp.A[j][l][1,1]) if q >= p]
                    Threads.@threads for q in qs
                        Arblib.add!(S[j][p,q],S[j][p,q], dot(tempX[2].blocks[j].blocks[l], sdp.A[j][l][1,1][q]))
                    end
                end
            else # low rank case

                # We basically have three options for using the cholesky factors
                # 1) form the full matrix corresponding to rightvecs and left vecs and do triangular solves and matrix multiplications.
                # Con: due to the block structure we can do it faster
                # pro: relatively easy
                # 2) Use the block structure to do smaller triangular solves/matmuls (i.e., similar to how we do it now)
                # con: more difficult to implement due to 'recursion' for the non-diagonal blocks
                # pro: we do have a block diagonal structure, so it'll save some time (probably)
                # 3) Explicitely invert X using the cholesky factor, and use that to do it like Y. 
                # We use method 3, since it doesn't seem to have much effect on the precision we need.
                # X_inv_block = similar(X_inv.blocks[j].blocks[l])
                Arblib.inv_cho_precomp!(tempX[1].blocks[j].blocks[l],X_inv.blocks[j].blocks[l])
                #if the matrices are not exact, Arblib.approx_mul! will copy the midpoints to a temporary matrix
                Arblib.get_mid!(tempX[1].blocks[j].blocks[l],tempX[1].blocks[j].blocks[l])

                for r=1:size(sdp.A[j][l],2)
                    Arblib.window_init!(part_r, temppart_r, 0, 0, size(Y.blocks[j].blocks[l],1), size(rightvecs[j][l][r],2))
                    # part_r = ArbRefMatrix(size(Y.blocks[j].blocks[l],1), size(rightvecs[j][l][r],2), prec=matmul_prec)
                    Arblib.window_init!(temp_window, Y.blocks[j].blocks[l], 0, (r-1)*delta, sz, r*delta)
                    matmul_threaded!(part_r, temp_window, rightvecs[j][l][r], prec=matmul_prec)
                    Arblib.window_clear!(temp_window)
                    Arblib.get_mid!(part_r, part_r)
                    for s=1:size(sdp.A[j][l],1)
                        Arblib.window_init!(temp_window, part_r, (s-1)*delta, 0, s*delta, size(part_r,2))
                        Arblib.window_init!(temp_window2, bilinear_pairings_Y[s,r], 0, 0, size(leftvecs[j][l][s],1), size(part_r,2))
                        matmul_threaded!(temp_window2, leftvecs[j][l][s], temp_window, prec=matmul_prec)#,opt=4)
                        Arblib.window_clear!(temp_window)
                        Arblib.window_clear!(temp_window2)
                        Arblib.get_mid!(bilinear_pairings_Y[s,r], bilinear_pairings_Y[s,r])
                    end
                    Arblib.window_init!(temp_window, tempX[1].blocks[j].blocks[l], 0, (r-1)*delta, sz, r*delta)
                    matmul_threaded!(part_r, temp_window, rightvecs[j][l][r], prec=matmul_prec)
                    Arblib.window_clear!(temp_window)
                    Arblib.get_mid!(part_r, part_r)
                    for s=1:size(sdp.A[j][l],1)
                        Arblib.window_init!(temp_window, part_r, (s-1)*delta, 0, s*delta, size(part_r,2))
                        Arblib.window_init!(temp_window2, bilinear_pairings_Xinv[s,r], 0, 0, size(leftvecs[j][l][s],1), size(part_r,2))
                        matmul_threaded!(temp_window2,leftvecs[j][l][s],temp_window,prec=matmul_prec)#,opt=4)
                        Arblib.window_clear!(temp_window)
                        Arblib.window_clear!(temp_window2)
                        Arblib.get_mid!(bilinear_pairings_Xinv[s,r],bilinear_pairings_Xinv[s,r])
                    end
                    Arblib.window_clear!(part_r)
                end

                #We collect the parts v^T Y v, because we need them to compute <A_*, Y>
                for r = 1:size(sdp.A[j][l],1)
                    for s = 1:r
                        Arblib.zero!(A_Y[j][l][r,s])
                        idx = 1
                        for p in keys(sdp.A[j][l][r,s])
                            for rnk=1:length(sdp.A[j][l][r,s][p].eigenvalues)
                                #This way we have A_Y in the order of [for p in Ps for i=1:rank]
                                # here we take the transpose element, since we only have the bilinear pairings for r <= s
                                # if r != s
                                #     A_Y[j][l][r,s][idx,1] = bilinear_pairings_Y[s,r][pointers_right[j][l][s][(r,p,rnk)], pointers_left[j][l][r][(s,p,rnk)]]
                                # else
                                A_Y[j][l][r,s][idx,1] = bilinear_pairings_Y[r,s][pointers_left[j][l][r][(s,p,rnk)], pointers_right[j][l][s][(r,p,rnk)]]
                                # end
                                # end
                                idx+=1
                            end
                        end
                    end
                end

                # We compute the contribution of this l to S[j]
                # The elements of S are per p,q, and then we need to sum over r,s,rnk  for both p,q
                # So we can loop over r1,s1, parallellize over p, loop over rank(p) r2,s2,q,rank(q)
                # and then add the contribution to S[j][p,q]
                for r1=1:size(sdp.A[j][l],1)
                    for s1=1:size(sdp.A[j][l],2)
                        # We need collect to combine keys(Dict()) with @threads, since @threads need an ordering and the keys are unordered
                        Threads.@threads for p in collect(keys(sdp.A[j][l][r1,s1]))
                            #NOTE: for high p, we do less q. So this may not be very well balanced. the order through keys may be kind of random, so this may be a better ordering than sorted
                            tot = Arb(prec=prec)
                            for r2=1:size(sdp.A[j][l],1)
                                for s2=1:size(sdp.A[j][l],2)
                                    for q in keys(sdp.A[j][l][r2,s2])
                                        # We can do only the upper triangular part here, so q >= p
                                        if q<p
                                            continue
                                        end
                                        for rnk1 = 1:length(sdp.A[j][l][r1,s1][p].eigenvalues)
                                            for rnk2 = 1:length(sdp.A[j][l][r2,s2][q].eigenvalues)
                                                #NOTE: we only have the bilinear pairings for r <= s. In the other case, we need to take the transpose element
                                                #S[j][p,q] += λ_p * λ_q * bilinearX * bilinearY.
                                                # The two comparisons do not add much to the time, but save 50% when both eigenvalues are 1.
                                                # Also, in general it is easy to get 1, just multiply one of the vectors by the eigenvalue. But that might give more unique eigenvectors
                                                if sdp.A[j][l][r1,s1][p].eigenvalues[rnk1] != sdp.A[j][l][r2,s2][q].eigenvalues[rnk2] || sdp.A[j][l][r2,s2][q].eigenvalues[rnk2] != 1
                                                    Arblib.mul!(tot,sdp.A[j][l][r1,s1][p].eigenvalues[rnk1],
                                                        sdp.A[j][l][r2,s2][q].eigenvalues[rnk2])
                                                    Arblib.mul!(tot,tot,bilinear_pairings_Xinv[s1,r2][pointers_left[j][l][s1][(r1,p,rnk1)],pointers_right[j][l][r2][(s2,q,rnk2)]])
                                                    Arblib.addmul!(S[j][p,q],tot,bilinear_pairings_Y[s2,r1][pointers_left[j][l][s2][(r2,q,rnk2)],pointers_right[j][l][r1][(s1,p,rnk1)]])
                                                else #both eigenvalues are 1, so we don't have to multiply by them
                                                    Arblib.addmul!(S[j][p,q],
                                                        bilinear_pairings_Xinv[s1,r2][pointers_left[j][l][s1][(r1,p,rnk1)],pointers_right[j][l][r2][(s2,q,rnk2)]],
                                                        bilinear_pairings_Y[s2,r1][pointers_left[j][l][s2][(r2,q,rnk2)],pointers_right[j][l][r1][(s1,p,rnk1)]])
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            # We call the gc when the memory usage gets too large. This is hardcoded, and the size we require might not be the best one.
            # When we use more than 5GB we call the GC. We could do it as percentage of the free or total memory.
            # That has the downside that it depends on the system state. 5GB hardcoded is also not a very good idea, especially for systems with a low amount of RAM
            if Sys.free_memory()/1024^3 < base - 5 # we used at least 5GB for the calculations
                GC.gc() #GC.gc(false) doesn't work very well when called frequently. i.e., it doesn't help a lot with freeing the memory on Linux
                malloc_trim(0)
            end
        end
        symmetric!(S[j])
        Arblib.get_mid!(S[j], S[j])
    end
    GC.gc(false) # clean up some of the garbage we created in this function, in case the matrices are small & GC.gc() doesn't get called
end

"""Compute the decomposition of [S B; B^T 0]"""
function compute_T_decomposition!(sdp,S,A_Y,X_inv, Y,bilinear_pairings_Y,bilinear_pairings_Xinv,LinvB,Q, tempX, threadinginfo,leftvecs,rightvecs,pointers_left,pointers_right,high_ranks, part_r; prec=precision(S[1]))
    # 1) pre compute bilinear basis
    # 2) compute S
    # 3) compute cholesky decomposition S = LL^T
    # 4) compute decomposition of [S B; B^T 0]

    # 1,2) compute the bilinear pairings and S, integrated. (per j,l, compute first pairings then S[j] part)
    time_schur = @elapsed begin
        compute_S_integrated!(S,sdp,A_Y, X_inv, Y,bilinear_pairings_Y,bilinear_pairings_Xinv,leftvecs,rightvecs,pointers_left,pointers_right,high_ranks,tempX, part_r, matmul_prec=prec)
    end

    #3) cholesky decomposition of S
    # We use a floating point version for the Cholesky, instead of the version of Arblib. It does exactly the same, but zeroes the error bounds during the algorithm
    time_cholS = @elapsed begin
        Threads.@threads for j in threadinginfo.j_order
            succes = approx_cholesky!(S[j])
            Arblib.get_mid!(S[j],S[j])
            if succes == 0
                throw(SolverFailure("S was not decomposed succesfully, try again with higher precision"))
            end
        end
    end

    #4) compute decomposition:
    #L^-1B
    time_LinvB = @elapsed begin
        Threads.@threads for j in threadinginfo.j_order
            Arblib.approx_solve_tril!(LinvB[j], S[j], sdp.B[j], 0)
            Arblib.get_mid!(LinvB[j],LinvB[j])
        end
    end

    #compute Q = B^T L^-T L^-1 B
    time_Q = @elapsed begin
        #options: 
        # vcat(LinvB...) and transpose -> 2x size of LinvB extra (option: preallocate and reuse every iteration)
        # per thread a copy of Q, Q_copy += LinvB[j]' * LinvB[j] -> 1x size of LinvB extra, 2* n_threads * size Q extra
        total_LinvB::ArbRefMatrix = vcat(LinvB...)
        matmul_threaded!(Q,transpose(total_LinvB),total_LinvB)
        Arblib.get_mid!(Q,Q)
    end

    time_cholQ = @elapsed begin
        succes = approx_cholesky!(Q)
        Arblib.get_mid!(Q,Q)
        if succes == 0
            throw(SolverFailure("Q was not decomposed correctly. Try restarting with a higher precision."))
        end
    end

    #all matrices are preallocated and modified, so we don't have to return them
    return time_schur,
    time_cholS,
    time_LinvB,
    time_Q,
    time_cholQ
end

"""Compute the vector <A_*,Z> = Tr(A_* Z) for one or all constraints"""
function trace_A(sdp, Z::BlockDiagonal,vecs_left,vecs_right,high_ranks)
    #Assumption: Z is symmetric
    result = ArbRefMatrix(sum(size.(sdp.c,1)), 1, prec = precision(Z))
    Arblib.zero!(result)
    #result has one entry for each (j,p) tuple
    res = Arb(prec=precision(Z)) #we use this every iteration, but we set it to zero when needed
    j_idx = 0 #the offset of the constraint index depending on the cluster
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            #if A[j][l] is of high rank, we can just do the normal matrix inner product
            delta = div(size(Z.blocks[j].blocks[l],1),size(sdp.A[j][l],1))
            if high_ranks[j][l]
                #high rank matrices only have 1 subblock
                for p in keys(sdp.A[j][l][1,1])
                    Arblib.add!(result[j_idx + p], dot(Z.blocks[j].blocks[l], sdp.A[j][l][1,1][p]),result[j_idx + p])  
                end
            else
                ones = ArbRefMatrix(1,delta,prec=precision(Z))
                Arblib.ones!(ones)
                for r=1:size(sdp.A[j][l],1)
                    for s=1:r
                        # Approach of Simmons Duffin in SDPB: (Z[r,s] V_r) ∘ V_l (where ∘ is the entrywise (Hadamard) product, and V is the matrix of all vectors as columns
                        # Then multiply with the vector of all ones to add the rows together for the dot products.
                        # Then we need to add this to the block of result corresponding to j
                        # However, we need to take care of different ranks. So we can calculate it like this, but we cannot put the result just in the vector because we have low rank instead of rank 1

                        #Here we make sure that we distribute all samples exactly once over the threads
                        #Note that this is slightly different from normal threaded matrix multiplication because we also do an entrywise product
                        used_threads = Threads.nthreads()
                        min_thread_size = div(size(vecs_right[j][l][r,s],2), used_threads)
                        # we add 1 to the first k threads such that all columns are being used
                        thread_sizes = [min_thread_size + (used_threads*min_thread_size + i <= size(vecs_right[j][l][r,s],2) ? 1 : 0) for i=1:used_threads]
                        indices = [0, cumsum(thread_sizes)...]
                        result_parts = [ArbRefMatrix(1,indices[i+1]-indices[i],prec=precision(Z)) for i=1:used_threads]

                        #apply the matrix multiplications: ones * (V_l o (Z * V_r))
                        Threads.@threads for i=1:used_threads
                            #window matrices
                            w1 = ArbRefMatrix(0,0)
                            w2 = ArbRefMatrix(0,0)
                            Arblib.window_init!(w1, Z.blocks[j].blocks[l], (r-1)*delta, (s-1)*delta, r*delta, s*delta)
                            Arblib.window_init!(w2,vecs_right[j][l][r,s], 0, indices[i], size(vecs_right[j][l][r,s],1), indices[i+1])
                            ZV = ArbRefMatrix(delta,indices[i+1]-indices[i],prec=precision(Z))
                            # we can parallellize here over the samples (rows of vs_transpose)
                            Arblib.approx_mul!(ZV,w1,w2)
                            Arblib.window_clear!(w1)
                            Arblib.window_clear!(w2)

                            Arblib.window_init!(w1, vecs_left[j][l][r,s], 0, indices[i], size(vecs_left[j][l][r,s],1), indices[i+1])
                            Arblib.mul_entrywise!(ZV,ZV,w1)
                            Arblib.window_clear!(w1)
                            Arblib.approx_mul!(result_parts[i],ones,ZV)
                        end
                        #Because we did the multiplications in this order we have row vectors to concatenate
                        result_part = hcat(result_parts...)

                        #now we add the (j,p,r,s,rank) parts to the (j,p) entries, multiplied by the eigenvalues
                        idx = 1
                        for p in keys(sdp.A[j][l][r,s])
                            Arblib.zero!(res)
                            for rnk=1:length(sdp.A[j][l][r,s][p].rightevs)
                                Arblib.addmul!(res,sdp.A[j][l][r,s][p].eigenvalues[rnk],result_part[1,idx])
                                idx+=1
                            end
                            if r != s #We need to take both sides of the matrix into account
                                Arblib.mul!(res,res,2)
                            end
                            Arblib.add!(result[j_idx+p,1],res,result[j_idx+p,1])
                        end
                    end
                end
            end
        end
        j_idx += size(sdp.c[j],1)
    end
    return result
end

function trace_A(sdp, (Y, A_Y), leftvecs_pairings,rightvecs_pairings,high_ranks)
    #Here we have precomputed v^T A v already, so we dont need the vectors. But it also doesnt cost extra time, so for ease of programming we allow them
    # So what we still need to do is get the right entries from A_Y, multiply them by the right eigenvalues and sum them to get entries corresponding to (j,p)
    result = ArbRefMatrix(sum(size.(sdp.c,1)), 1, prec = precision(sdp.b))
    #we can parallellize over the constraints because we calculate the index. For clusters with few constraints this might be slower than not parallelizing
    j_idx = 0
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            #for high rank matrices A[j][l], we need to take a standard inner product.
            if high_ranks[j][l]
                #high rank matrices have only one subblock
                for p in keys(sdp.A[j][l][1,1])
                    Arblib.add!(result[j_idx + p], dot(Y.blocks[j].blocks[l], sdp.A[j][l][1,1][p]),result[j_idx + p])  
                end
            else
                for r=1:size(sdp.A[j][l],1)
                    for s=1:r
                        # index_to_prnk = [(p,rnk) for p in keys(sdp.A[j][l][r,s]) for rnk=1:length(sdp.A[j][l][r,s][p].eigenvalues)]
                        # prnk_to_index = Dict((p,rnk)=>i for (i,(p,rnk)) in enumerate(index_to_prnk))
                        idx = 1
                        for p in collect(keys(sdp.A[j][l][r,s])) # We need collect to use @threads with keys(Dict())
                            tot = Arb(0,prec=precision(sdp.b))
                            for rnk=1:length(sdp.A[j][l][r,s][p].eigenvalues)
                                #contribution for this p is
                                Arblib.addmul!(tot,A_Y[j][l][r,s][idx,1],sdp.A[j][l][r,s][p].eigenvalues[rnk])
                                idx+=1
                            end
                            if r!=s #We take both s,r and r,s into account
                                Arblib.mul!(tot,tot,2)
                            end
                            Arblib.add!(result[j_idx+p,1],tot,result[j_idx+p,1])
                        end
                    end
                end
            end
        end
        j_idx += size(sdp.c[j],1)
    end
    return result
end

"""Set initial_matrix to ∑_i a_i A_i"""
function compute_weighted_A!(initial_matrix, sdp, a,vecs_left,high_ranks)
    # initial_matrix is a BlockDiagonal matrix of BlockDiagonal matrices of ArbMatrices
    # We add the contributions to the (blocked) upper triangular, then symmetrize
    Q = ArbRefMatrix(0,0,prec=precision(a))
    w = ArbRefMatrix(0,0,prec=precision(a))
    j_idx = 0
    for j in eachindex(sdp.A)
        for l in eachindex(sdp.A[j])
            Arblib.zero!(initial_matrix.blocks[j].blocks[l])
            #assuming all subblocks have the same size: (If we ever want to change that: make Q inside the r,s loops since only then we will know the size)
            #Here we again need to make a distinction between low and high rank matrices A
            # High rank is easy: we can just add x_p * A_p
            if high_ranks[j][l]
                #high rank matrices only have one subblock
                for p in keys(sdp.A[j][l][1,1])
                    Arblib.addmul!(initial_matrix.blocks[j].blocks[l], sdp.A[j][l][1,1][p], a[j_idx+p,1])
                end
            else              
                delta = div(size(initial_matrix.blocks[j].blocks[l],1),size(sdp.A[j][l],1))
                #initialize the scratch spaces
                cur = Arb(1,prec=precision(a))
                vec = ArbRefMatrix(delta,1,prec=precision(a))
                
                for r = 1:size(sdp.A[j][l],1)
                    for s = 1:r
                        Arblib.window_init!(Q, initial_matrix.blocks[j].blocks[l],(r-1)*delta, (s-1)*delta, r*delta, s*delta)
                        # Approach: sum_i a_i A_i can be written as V_i D V_i^T, with D diagonal (λ_i,rnk * a_i)
                        # Now we compute V_r D
                        # Scratch space:
                        # Note that vecs_right has the same number of vectors as vecs_left, of the same size, because A[j][l][r,s] has the same number of leftevs as rightevs
                        vecs_right = ArbRefMatrix(size(vecs_left[j][l][r,s],2), size(vecs_left[j][l][r,s],1),prec=precision(a)) #V_rD. So we multiply every vector (column) by the corresponding eigenvalue
                        idx = 1
                        for p in keys(sdp.A[j][l][r,s])
                            for rnk in eachindex(sdp.A[j][l][r,s][p].eigenvalues)
                                Arblib.mul!(cur,a[j_idx+p,1],sdp.A[j][l][r,s][p].eigenvalues[rnk])
                                Arblib.mul!(vec, sdp.A[j][l][r,s][p].rightevs[rnk],cur)
                                Arblib.window_init!(w, vecs_right, idx-1, 0,idx, delta)
                                Arblib.transpose!(w, vec)
                                Arblib.window_clear!(w)
                                idx+=1
                            end
                        end
                        Arblib.get_mid!(vecs_right,vecs_right)

                        # calculate V_rD * V_l^T
                        matmul_threaded!(Q,vecs_left[j][l][r,s], vecs_right)
                        Arblib.transpose!(Q,Q)
                        Arblib.get_mid!(Q,Q)
                        Arblib.window_clear!(Q)
                    end
                end
            end
            if size(sdp.A[j][l],1)>1
                #We only have to symmetrize when there are offdiagonal blocks (i.e. r!=s)
                symmetric!(initial_matrix.blocks[j].blocks[l],:L) #Takes the lower part
            end
        end
        j_idx += size(sdp.c[j],1)
    end
    return nothing #initial_matrix is modified so we return nothing
end


"""Compute the search directions, using a precomputed decomposition"""
function compute_search_direction!(
    sdp,
    dx, dX, dy, dY,  #these are modified
    P,
    p,
    d,
    R,
    X_inv,
    Y,
    tempX,
    threadinginfo,
    L,
    LinvB,
    Q,
    leftvecs_pairings,
    rightvecs_pairings, 
    high_ranks,
)
    prec = precision(Y)
    # using the decomposition, compute the search directions
    # 5) solve system with rhs -d - <A_*, Z>) for dx and rhs p for dy, where Z = X^{-1}(PY - R)
    # 6) compute dX = P + sum_i A_i dx_i
    # 7) compute dY = X^{-1}(R-dX Y) (XdY = R-dXY)
    # 8) symmetrize dY = 1/2 (dY +dY')
    time_Z = @elapsed begin
        #We use dY as Z. We can use dX as scratch space here, since we calculate the new dX and dY in this function
        Threads.@threads for (j,l) in threadinginfo.jl_order
            #Z = X_inv*(P*Y-R)
            #We use Z (dY) as scratch space; Z = X_inv ((PY)-R)
            #Note that dY is overwritten in the first multiplication, hence we don't have to zero it.
            Arblib.approx_mul!(dY.blocks[j].blocks[l], P.blocks[j].blocks[l], Y.blocks[j].blocks[l])
            Arblib.sub!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l], R.blocks[j].blocks[l])
            Arblib.solve_cho_precomp!(dY.blocks[j].blocks[l], X_inv.blocks[j].blocks[l], dY.blocks[j].blocks[l])
            #We symmetrize Z in order to use <A_*,Z> correctly (because when m[j][l]>1, we calculate contributions based on the upper triangular block)
            Arblib.transpose!(tempX[1].blocks[j].blocks[l],dY.blocks[j].blocks[l]) #we use dX as scratch space for the transpose
            Arblib.add!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l], tempX[1].blocks[j].blocks[l])
            Arblib.div!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l], 2)
            #remove the error bounds
            Arblib.get_mid!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l])
        end
    end

    #the right hand sides of the system
    rhs_y = p
    time_rhs_x = @elapsed begin
        #rhs_x = -d - <A_*,Z>
        #we use dx for rhs_x
        Arblib.neg!(dx,d)
        Arblib.sub!(dx,dx,trace_A(sdp, dY,leftvecs_pairings,rightvecs_pairings,high_ranks))
        Arblib.get_mid!(dx, dx)
    end

    # solve the system (L 0; B^TL^-T I)(I 0; 0 Q_L Q_L^T)(L^T -L^-1B; 0 I)(dx; dy) = (rhs_x; rhs_y)
    indices = [0, cumsum(size.(sdp.c,1))...] #0, P[1], P[1]+P[2],... ,sum(P)
    time_sys = @elapsed begin
        temp_x = [
            ArbRefMatrix(indices[j+1] - indices[j], 1, prec = prec) for
            j in eachindex(sdp.A)
        ]
        temp_y = [similar(rhs_y) for j in eachindex(sdp.A)]
        #The first lower triangular system: Lx = rhs_x, blockwise
        #y = rhs_y - B^TL^-T x
        Threads.@threads for j in threadinginfo.j_order
            Arblib.approx_solve_tril!(
                temp_x[j],
                L[j],
                dx[indices[j]+1:indices[j+1], :],
                0,
            )
            Arblib.get_mid!(temp_x[j],temp_x[j])
            # Arblib.approx_mul!(temp_y[j], transpose(LinvB[j]), temp_x[j])
            approx_mul_transpose!(temp_y[j], LinvB[j], temp_x[j])
        end

        # dy = rhs_y - ∑_j temp_y[j]
        Arblib.set!(dy,rhs_y)
        for y in temp_y
            Arblib.sub!(dy,dy,y)
        end
        Arblib.get_mid!(dy,dy)

        #second system: dy = Q^-1 dy, dx stays the same
        Arblib.solve_cho_precomp!(dy,Q,dy)
        Arblib.get_mid!(dy,dy)

        #third system: dy stays the same, Udx = dx_old + LinvB dy
        dx_perj = [
            ArbRefMatrix(indices[j+1] - indices[j], 1, prec = prec) for
            j in eachindex(sdp.A)
        ]
        #dx[j] = L[j]^-1 (dx_old[j]+L^-1B[j] dy)
        Threads.@threads for j in threadinginfo.j_order
            Arblib.transpose!(L[j],L[j])
            Arblib.approx_mul!(dx_perj[j],LinvB[j],dy)
            Arblib.add!(temp_x[j],temp_x[j],dx_perj[j])
            Arblib.get_mid!(temp_x[j],temp_x[j])
            Arblib.approx_solve_triu!(dx_perj[j], L[j], temp_x[j], 0)
            Arblib.transpose!(L[j],L[j]) #We transpose again because we need L[j] both for the predictor and the corrector search direction
        end

        # Put the dx parts into the dx vector:
        cur_idx = 0
        for j in eachindex(dx_perj)
            dx[cur_idx+1:cur_idx+size(dx_perj[j],1),1] = dx_perj[j]
            cur_idx += size(dx_perj[j],1)
        end
        Arblib.get_mid!(dx,dx)
    end #of timing system

    #step 6:
    time_dX = @elapsed begin
        #dX = ∑_i dx_i A_i + P
        #compute the sum
        compute_weighted_A!(dX, sdp, dx,leftvecs_pairings,high_ranks)
        #add P
        Threads.@threads for (j,l) in threadinginfo.jl_order
            Arblib.add!(dX.blocks[j].blocks[l],dX.blocks[j].blocks[l],P.blocks[j].blocks[l])
            Arblib.get_mid!(dX.blocks[j].blocks[l], dX.blocks[j].blocks[l])
        end
    end

    #step 7 & 8: compute dY and symmetrize
    time_dY = @elapsed begin
        Threads.@threads for (j,l) in threadinginfo.jl_order
            #dY = X_inv * (R- dX *Y) = X_inv * ( R- (dX*Y))
            Arblib.zero!(dY.blocks[j].blocks[l])
            #Again, we use dY as scratch space, working from inner brackets to outer brackets.
            #Note that every block of dY is overwritten in the first multiplication. Hence we don't have to zero it
            Arblib.approx_mul!(dY.blocks[j].blocks[l], dX.blocks[j].blocks[l], Y.blocks[j].blocks[l])
            Arblib.sub!(dY.blocks[j].blocks[l], R.blocks[j].blocks[l], dY.blocks[j].blocks[l])
            Arblib.solve_cho_precomp!(dY.blocks[j].blocks[l], X_inv.blocks[j].blocks[l], dY.blocks[j].blocks[l])
            #Symmetrize dY. Since we don't have extra scratch space, we need to create a new matrix for the transpose (implicitely)
            Arblib.transpose!(tempX[1].blocks[j].blocks[l], dY.blocks[j].blocks[l])
            Arblib.add!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l], tempX[1].blocks[j].blocks[l])
            Arblib.div!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l], 2)
            #remove the error bounds
            Arblib.get_mid!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l])
        end
    end

    return time_Z, time_rhs_x, time_sys, time_dX, time_dY
end

"""Compute the step length min(γ α(M,dM), 1), where α is the maximum number step
to which keeps M+α(M,dM) dM positive semidefinite"""
function compute_step_length(
    M::BlockDiagonal,
    dM::BlockDiagonal,
    gamma,
    tempX,
    threadinginfo,
)
    # M+ α dM is PSD iff I + α L^-1 dM L^-T is PSD iff α < - 1/eigmin(L^-1 dM L^-T).
    # We take α = - γ/eigmin(L^-1 dM L^-T)
    prec = precision(M) #*2 #test if increasing the precision in this part of the code helps for the rest
    # We parallellize over j and l, but need the total minimum. We save the minimum for every (j,l)
    # We actually just need to keep track of the minimum for every thread, but it doesn't matter much
    min_eig_arb = [[Arb(Inf) for l in eachindex(M.blocks[j].blocks)] for j in eachindex(M.blocks)]
    Threads.@threads for (j,l) in threadinginfo.jl_order
        if size(M.blocks[j].blocks[l],1) == 1
            # M_j,l is just a 1x1 matrix
            # So L^-1 dM L^-T = dM/M
            Arblib.div!(min_eig_arb[j][l], dM.blocks[j].blocks[l][1,1],M.blocks[j].blocks[l][1,1])
            continue
        end
        # chol = similar(M.blocks[j].blocks[l])
        succes = approx_cholesky!(tempX[1].blocks[j].blocks[l], M.blocks[j].blocks[l])
        if succes == 0
            throw(SolverFailure("The cholesky decomposition could not be computed during the computation of the step length. Please try again with a higher precision."))
        else
            # Arblib.get_mid!(chol,chol)
            # LML = similar(dM.blocks[j].blocks[l])
            #tempX[2].blocks[j].blocks[l] = chol^-1 dMblock
            Arblib.approx_solve_tril!(tempX[2].blocks[j].blocks[l], tempX[1].blocks[j].blocks[l], dM.blocks[j].blocks[l], 0)
            Arblib.transpose!(tempX[2].blocks[j].blocks[l], tempX[2].blocks[j].blocks[l])
            Arblib.get_mid!(tempX[2].blocks[j].blocks[l],tempX[2].blocks[j].blocks[l])
            #temp tempX[2].blocks[j].blocks[l] = chol^-1 (chol^-1 dMblock)^T
            Arblib.approx_solve_tril!(tempX[2].blocks[j].blocks[l], tempX[1].blocks[j].blocks[l], tempX[2].blocks[j].blocks[l], 0)
            #because we can usually use floating point numbers for this, we use a julia implementation of the Lanczos method in KrylovKit
            #TODO: extra check that the converged eigenvalue/vector pair is also converged in high precision?
            values, vecs, info = eigsolve(Float64.(tempX[2].blocks[j].blocks[l]), 1, :SR; krylovdim = 10, maxiter = min(100,size(tempX[2].blocks[j].blocks[l],1)),tol=10^-5, issymmetric=true, eager=true)

            if info.converged >= 1
                min_eig_arb[j][l] = Arb(values[1],prec=prec) - 10^-5 #tolerance, so this surely is smaller than the minimum eigenvector
                continue
            end

            eigenvalues = AcbRefVector(size(tempX[2].blocks[j].blocks[l], 1),prec=prec)
            #converting to AcbMatrix allocates, but we cannot avoid it because this function is only available for AcbMatrices
            # We need to do it somewhere, so we can as well use the faster Arb stuff for the previous operations
            A = AcbRefMatrix(tempX[2].blocks[j].blocks[l],prec=prec)
            succes2 = Arblib.approx_eig_qr!(eigenvalues,A,prec=prec)
            if succes2 == 0 #even in this case it is possible that the output is accurate enough. But that is difficult to detect. Anyway, the cholesky for S or Q usually fail first
                throw(SolverFailure("The eigenvalues could not be computed during the computation of the step length. Please try again with a higher precision."))
            else
                real_arb = Arb(0, prec = prec)
                for i = 1:length(eigenvalues)
                    Arblib.get_real!(real_arb, eigenvalues[i])
                    Arblib.min!(min_eig_arb[j][l],real_arb,min_eig_arb[j][l])
                end
            end
        end
    end

    #Get the minimim over all blocks:
    eigs = [BigFloat(min_eig_arb[j][l]) for j in eachindex(M.blocks) for l in eachindex(M.blocks[j].blocks)]
    # println(eigs)
    min_eig = minimum(eigs)


    if min_eig > -gamma # 1 is the maximum step length
        return Arb(1, prec = prec)
    else
        return Arb(-gamma / min_eig, prec = prec)
    end
end
