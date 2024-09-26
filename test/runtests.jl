using ClusteredLowRankSolver, Nemo
using AbstractAlgebra: RealField
using Test

@testset "ClusteredLowRankSolver.jl" begin

    @testset "examples" begin
        # these examples test nearly everything
        include("../examples/PolyOpt.jl")
        using  .PolyOpt
        problem, _, dualsol = min_f(2)
        @test  objvalue(problem, dualsol) ≈ -2.113 atol=1e-2

        include("../examples/Delsarte.jl")
        using .Delsarte
        @test delsarte(3, 10, 1//2) ≈ 13.158314 atol=1e-5

        include("../examples/SpherePacking.jl")
        using .SpherePacking
        problem, _, dualsol = cohnelkies(8, 15, prec=256)
        @test objvalue(problem, dualsol) ≈ BigFloat(pi)^4/384 atol=1e-4 #exact in the limit of d-> ∞, but for this d the error still is relatively large
        problem, _, dualsol = Nsphere_packing(8, 15, [1//2,1//2],2, prec=512)
        @test objvalue(problem, dualsol) ≈ BigFloat(pi)^4/384 atol=1e-4

        include("../examples/ThreePointBound.jl")
        using .ThreePointBound
        problem, _, dualsol = three_point_spherical_codes(4, 1//6, -1, 4, prec=256, omega_d=10^3, omega_p=10^3)
        @test objvalue(problem, dualsol) ≈ 10 atol=1e-5
    end

    @testset "modelling" begin
        obj = Objective(0, Dict(:z=> hcat([1])), Dict())
        constraint = Constraint(1,Dict(:z=>hcat([1]),:z2=>hcat([1])), Dict())
        oldproblem = Problem(Maximize(obj), [constraint])
        newproblem = model_psd_variables_as_free_variables(oldproblem, [:z])
        _,_,dualsol1,_ = solvesdp(oldproblem)
        _,_,dualsol2,_ = solvesdp(newproblem)
        @test objvalue(oldproblem, dualsol1) ≈ objvalue(newproblem, dualsol2) atol=1e-10
    end

    @testset "warnings" begin
        obj = Objective(0, Dict(:z=> hcat([1])), Dict())
        constraint = Constraint(1,Dict(:z=>hcat([1]),:z2=>hcat([1])), Dict())
        c2 = Constraint(1, Dict(:z=>LowRankMatPol([1],[[1]])), Dict())
        c3 = Constraint(1, Dict(Block(:z2)=>hcat([1])), Dict())
        problem = Problem(Maximize(obj), [constraint, c2])
        problem2 = Problem(Maximize(obj), [constraint, c3])
        @test_logs (:warn, "Please use LowRankMatPol consistently for the constraint matrices corresponding to the variable z. Converting to normal matrices.") ClusteredLowRankSDP(problem)
        @test_logs (:warn, "Please use Block consistently. Solutions with Block(z2) will be returned.") ClusteredLowRankSDP(problem2)
    end

    @testset "Rounding" begin
        include("../examples/DelsarteExact.jl")
        using .DelsarteExact
        @test begin
            success, problem, exactdualsol = delsarte_round(8, 3, 1//2)
            success && objvalue(problem, exactdualsol) == 240           
        end
        @test begin
            problem, primalsol, dualsol = three_point_spherical_codes(4, 1//6, -1, 4,prec=256, duality_gap_threshold=1e-30, omega_d = 10^3, omega_p=10^3)
            success, exactdualsol = exact_solution(problem, primalsol, dualsol)
            success && objvalue(problem, exactdualsol) == 10
        end
        # rounding over a different field
        R, x = polynomial_ring(QQ, :x)
        N, z = number_field(x^2 - 5, :z)
        gapprox = sqrt(big(5))
        obj, problem, primalsol, dualsol = delsarte_exact(4, 9, 1/(z-1); FF=N, g = gapprox)
        @test begin
            N2, gapprox2 = find_field(primalsol,dualsol)
            # check that it is the same field
            ginfield = to_field(gapprox, N2, gapprox2)
            gapprox3 = generic_embedding(ginfield, gapprox2)
            abs(gapprox3 - gapprox) < 1e-10 
        end 
        @test begin
            # round the approximate solution to an exact solution
            success, exactdualsol = exact_solution(problem, primalsol, dualsol; FF=N, g=gapprox)
            objexact = objvalue(problem, exactdualsol)
            success && objexact == 120
        end
    end


    @testset "SampledMPolyElem" begin #this is mostly tested through the examples too
        R, (x,) = polynomial_ring(RealField, ["x"])
        p1 = x^2 + 2
        samples = [[i] for i=0:10]
        Rsampled = sampled_polynomial_ring(BigFloat, samples)
        p2 = Rsampled(p1)
        @testset "addition" begin
            @test (p1+p2)(5) == 2*p1(5)
            @test (p2+p1)(5) == 2*p1(5)
            @test (p2+p2)(5) == 2*p1(5)
        end
        @testset "subtraction" begin
            @test (p1-p2)(4) == 0
            @test (p2-p1)(4) == p2(4)-p1(4)
            @test (p2-p2)(3) == 0
        end
        @testset "multiplication" begin
            @test (p1*p2)(5) == p1(5)^2
            @test (p2*p1)(5) == p1(5)^2
            @test (p2*p2)(5) == p1(5)^2
        end
        @testset "substitution" begin
            # multivariate substitution
            p1(p2)(1) == 11
            for FF in [QQ, ZZ]
                R2, x2 = polynomial_ring(FF, [:x])
                q = x2[1]^2+1
                Rsampled2 = sampled_polynomial_ring(FF, [[i] for i=0:10])
                q2 = Rsampled2(q)
                @test q(q2)(FF(1)) == 5
            end
            # univariate substitution
            for FF in [QQ, ZZ]
                R2, x2 = polynomial_ring(FF, :x)
                q = x2^2+1
                Rsampled2 = sampled_polynomial_ring(FF, collect(0:10))
                q2 = Rsampled2(q)
                @test q(q2)(FF(1)) == 5
            end
        end
    end

    @testset "LowRankMat(Pol)" begin
        R, x = polynomial_ring(RealField, "x")
        A = LowRankMatPol([x],[[x^2,x^3]]) # the matrix [x^5 x^6; x^6 x^7]
        B = A(2)
        @test [B[i,j] for i=1:2,j=1:2] == [2^5 2^6; 2^6 2^7]
        At = transpose(A)
        Bt = At(2)
        @test [Bt[i,j] for i=1:2,j=1:2] == [2^5 2^6; 2^6 2^7]
    end

    @testset "SDPA format" begin
        problem = sdpa_sparse_to_problem("example.dat-s", T=BigFloat)
        @test matrixcoeff(objective(problem), 2)[1,1] isa BigFloat
        @test matrixcoeff(constraints(problem)[2], 2)[1,1] isa BigFloat
        @test matrixcoeff(objective(problem), 2) == [3 0; 0 4]
        @test matrixcoeff(constraints(problem)[2], 2) == [5 2; 2 6]
    end
    @testset "checking" begin
        problem = sdpa_sparse_to_problem("example.dat-s")
        @test check_problem(problem)
        sdp = ClusteredLowRankSDP(problem)
        @test check_sdp!(sdp)
    end
end
