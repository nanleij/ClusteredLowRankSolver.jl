#functions to convert the format from SDPA sparse format to a Problem (or ClusteredLowRankSDP)

function read_sdpa_sparse_file(filename; T=Float64)
    lines = [split(x) for x in readlines(filename)]
    i = 1
    while !occursin(lines[i][1][1], "0123456789")
        i+=1
    end
    # @show lines[i]
    m = parse(Int, lines[i][1]); i+=1
    n_blocks = parse(Int, lines[i][1]); i+=1
    blocksizes = [parse(Int, x) for x in lines[i]]; i+=1
    diag_blocks = [idx for idx in eachindex(blocksizes) if blocksizes[idx] < 0]
    c = [parse(T, x) for x in lines[i]]; i+=1
    @assert length(c) == m
    blocks = [make_blocks(blocksizes; T=T) for j=0:m]
    for l in lines[i:end]
        cidx, bidx,i,j = [parse(Int,x) for x in l[1:4]]
        v = parse(T,l[5])
        if bidx in diag_blocks
            @assert i == j
            blocks[cidx+1][bidx][i][1,1] = v
        else
            blocks[cidx+1][bidx][i,j] = v
            blocks[cidx+1][bidx][j,i] = v
        end
    end

    return m, blocksizes, c, blocks
end

function make_blocks(blocksizes; T=Float64)
    blocks = []
    for (i,b) in enumerate(blocksizes)
        if b < 0
            push!(blocks, [zeros(T,1,1) for _=1:-b])
        else
            push!(blocks, zeros(T, b,b))
        end
    end 
    return blocks
    # return [zeros(Float64,abs(b),abs(b)) for b in blocksizes ]
end

"""
    sdpa_sparse_to_problem(filename,obj_shift = 0; T=Float64)

Define the `Problem` from the file `filename` assuming it is in SDPA sparse format,
using the number type `T`. Optionally add an objective shift. 
"""
function sdpa_sparse_to_problem(filename, obj_shift=0; T=Float64)
    m,blocksizes,c,blocks = read_sdpa_sparse_file(filename; T=T)
    dicts = [Dict() for k=0:m] #objective = 0, constraints are the rest
    for c_idx=1:m+1
        for (b_idx, b) in enumerate(blocksizes)
            if b < 0
                for b2_idx=1:-b
                    if !all( blocks[c_idx][b_idx][b2_idx] .== 0)
                        dicts[c_idx][(b_idx,b2_idx)] = blocks[c_idx][b_idx][b2_idx]
                    end
                end
            elseif !all( blocks[c_idx][b_idx] .== 0)
                dicts[c_idx][b_idx] = blocks[c_idx][b_idx]
            end
        end
    end
    obj = Objective(obj_shift,dicts[1],Dict())

    # Checking for and removing empty constraints
    removing = Int[]
    for i in eachindex(c)
        if length(dicts[i+1]) == 0 
            push!(removing, i)
            if c[i] != 0
                @warn "Constraint without constraint matrices but with nonzero constant found. Removing the constraint."
            else
                @info "Empty constraint found and removed."
            end
        end
    end
    cons = [Constraint(c[i], dicts[i+1], Dict()) for i in eachindex(c) if !(i in removing)]
    Problem(Maximize(obj), cons)
end

