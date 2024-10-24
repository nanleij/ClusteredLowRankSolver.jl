using PrecompileTools: @setup_workload, @compile_workload

@setup_workload let
    R, (x,) = polynomial_ring(QQ, 1)
    @compile_workload begin
        Rs = sampled_polynomial_ring(QQ, [k//3 for k=-2:2])
        
        c = [Constraint((1+x+x^2)^2, Dict(:A=>LowRankMatPol([1], [Rs.([1,x,x^2])])), Dict(:b=>1, :c=>x), [k//3 for k=-2:2])]
        o = Objective(0, Dict(:A => ones(Int, 3,3)), Dict())
        problem = Problem(Minimize(o), c)
        redirect_stdout(devnull) do
            solvesdp(problem)
        end
    end
end