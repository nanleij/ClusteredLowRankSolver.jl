using ClusteredLowRankSolver
using Test

@testset "ClusteredLowRankSolver.jl" begin

    @testset "examples" begin
        # these examples test nearly everything
        include("../examples/PolyOpt.jl")
        using  .PolyOpt
        _,sol,_,_ = min_f(2)
        @test sol.dual_objective ≈ -2.113 atol=1e-2

        include("../examples/Delsarte.jl")
        using .Delsarte
        _, sol, _, _ = delsarte(3, 10, 1//2)
        @test sol.dual_objective ≈ 13.158314 atol=1e-5
        _, sol, _, _ = delsarte(8, 10, 1//2, all_free=true)
        @test sol.dual_objective ≈ 240 atol = 1e-10

        _,sol, _,_ = delsarte_highrank(3,10,1//2)
        @test sol.dual_objective ≈ 13.158314 atol=1e-5


        include("../examples/SpherePacking.jl")
        using .SpherePacking
        _, sol, _, _ = cohnelkies(8, 15)
        @test sol.dual_objective ≈ BigFloat(pi)^4/384 atol=1e-4 #exact in the limit of d-> ∞, but for this d the error still is relatively large
        _, sol, _, _ = Nsphere_packing(8, 15, [1//2,1//2],2)
        @test sol.dual_objective ≈ BigFloat(pi)^4/384 atol=1e-4

        include("../examples/ThreePointBound.jl")
        using .ThreePointBound
        _, sol, _, _ = three_point_spherical_cap(3, 6, 1//2, 256)
        @test sol.dual_objective ≈ 12.718780 atol=1e-5
        _, sol, _, _ = three_point_spherical_cap(3, 6, 1//2, 256, reduce_memory = true)
        @test sol.dual_objective ≈ 12.718780 atol=1e-5
    end

    @testset "SampledMPolyElem" begin #this is mostly tested through the examples too
        using AbstractAlgebra
        R, (x,) = PolynomialRing(RealField, ["x"])
        p1 = x^2 + 2
        samples = [[i] for i=0:10]
        p2 = ClusteredLowRankSolver.SampledMPolyElem(p1,samples)
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
    end

    @testset "LowRankMat(Pol)" begin
        using AbstractAlgebra
        R, (x,) = PolynomialRing(RealField, ["x"])
        A = LowRankMatPol([x],[[x^2,x^3]]) # the matrix [x^5 x^6; x^6 x^7]
        B = A(2)
        @test [B[i,j] for i=1:2,j=1:2] == [2^5 2^6; 2^6 2^7]
        At = transpose(A)
        Bt = At(2)
        @test [Bt[i,j] for i=1:2,j=1:2] == [2^5 2^6; 2^6 2^7]
    end
end
