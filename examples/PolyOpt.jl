module PolyOpt

using ClusteredLowRankSolver, BasesAndSamples, AbstractAlgebra

export min_f

function invariant_basis(x,y,z, d)
    # create a vector with a precise type
    v = [(x*y*z)^0]
    for deg=1:d, j=0:div(deg,3), i=0:div(deg-3j,2)
        push!(v, (x+y+z)^(deg-2i-3j) * (x*y+y*z+z*x)^i * (x*y*z)^j)
    end
    return v
end

function min_f(d)
    obj = Objective(0, Dict(), Dict(:M => 1))

    FF = RealField
    R, (x,y,z) = polynomial_ring(FF, ["x", "y", "z"])
    # The polynomial f:
    f =  x^4 + y^4 + z^4 - 4x*y*z + x + y + z

    # An invariant basis up to degree d:
    basis = invariant_basis(x, y, z, 2d)
    # For the sum-of-squares polynomials we have to
    # select elements of the basis based on the degree
    degrees = [total_degree(p) for p in basis]

    # generate samples and a good basis
    cheb_points = [vcat(sample_points_chebyshev(2d+k)...) for k=0:2]
    samples_grid = [[cheb_points[1][i+1], cheb_points[2][j+1], cheb_points[3][k+1]]
        for i=0:2d for j=0:2d+1 for k=0:2d+2]
    basis, samples = approximatefekete(basis, samples_grid)

    psd_dict = Dict()
    symmetry_weights = [[[R(1)]],
                        [[R((x-y)*(y-z)*(z-x))]],
                        [[1/sqrt(FF(2))*(2x-y-z),1/sqrt(FF(2))*(2y*z-x*z-x*y)],
                            [sqrt(FF(3)/FF(2))*(y-z),sqrt(FF(3)/FF(2))*(x*z-x*y)]]]
    for swi=1:length(symmetry_weights)
        rank = length(symmetry_weights[swi])
        # This has in general too many entries, so we remove the ones with too high degree.
        vecs = [kron(symmetry_weights[swi][r], basis) for r=1:rank]
        for r=1:rank
            # the length of the symmetric basis we tensored the symmetric weight with
            l = length(basis)
            # we keep the elements with degree at most d
            keep_idx = [i for i=1:length(vecs[r]) if total_degree(symmetry_weights[swi][r][div(i-1,l)+1]) + degrees[(i-1)%l+1] <= d]
            vecs[r] = vecs[r][keep_idx]
        end
        vecs = [v for v in vecs if length(v) > 0]
        if length(vecs) > 0
            psd_dict[Block((:trivariatesos,swi))] = LowRankMatPol([R(1) for r=1:length(vecs)], vecs)
        end
    end

    constr = Constraint(f, psd_dict, Dict(:M => 1), samples)
    pol_problem = LowRankPolProblem(true, obj, [constr])
    sdp = ClusteredLowRankSDP(pol_problem)

    solvesdp(sdp)
end

end
