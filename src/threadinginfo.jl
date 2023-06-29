"""
Distribute the weights over n (about) equal-sized bins by swapping between bins with high total weight and low total weight.
"""
function distribute_weights_swapping(weights,n;nswaps = min(length(weights)^2,100))
    # we first 'normally' distribute the weights (first k to first thread, second k to second thread etc.)
    # Then we try to improve it a number of times by swapping weights between sets with a high and low total weight
    step = div(length(weights),n)+1 # first number of sets have this size, second part has this size-1
    nstep = n-(step*n-length(weights)) # this is the number of sets with size 'step'
    sets = vcat([collect((i-1)*step+1:i*step) for i=1:nstep],
            [collect(nstep*step+(i-1)*(step-1)+1:nstep*step+i*(step-1)) for i=1:n-nstep])
    set_weights = [sum(weights[sets[i]]) for i=1:length(sets)] #the total weights

    # edge case: less than 1 weight for every core
    if length(weights) <= n || n == 1
        return sets,set_weights,[weights[s] for s in sets]
    end

    index_set = 1
    index_el = 1
    for k=1:nswaps # nswaps is actually not the number of swaps, but the number of tries
        # The current (large) set and the current element of the set
        max_set = sort([(set_weights[i],i) for i=1:length(set_weights)],rev=true)[index_set][2]
        max_el = sets[max_set][sort([(weights[sets[max_set]][i],i) for i=1:length(weights[sets[max_set]])],rev=true)[index_el][2]]
        # The small set and the small element of the set
        min_set = argmin(set_weights)
        min_el = sets[min_set][argmin(weights[sets[min_set]])]
        #Determine whether the swap works: the min_set should not increase by too much, and the max should of course also not increase
        if set_weights[min_set]+weights[max_el]-weights[min_el] < set_weights[max_set] &&
            set_weights[max_set]-weights[max_el]+weights[min_el] < set_weights[max_set]
            # swapping distributes the weights more equally, so we do it

            # Make the new large set
            sets[max_set] = [i for i in sets[max_set] if i!= max_el]
            push!(sets[max_set],min_el)
            set_weights[max_set] += weights[min_el] - weights[max_el]

            #make the new small set
            sets[min_set] = [i for i in sets[min_set] if i!= min_el]
            push!(sets[min_set],max_el)
            set_weights[min_set] += weights[max_el] - weights[min_el]
            # We swapped, so we start looking again from the extreme sets
            index_el = 1
            index_set = 1
        elseif index_el < step-1 #some sets have 'step' elements, but others have 'step-1' elements. We don't look at which set we have but just be safe
            # We didn't swap, so we want to look at different elements in the next iteration
            index_el+=1
        elseif index_el == step-1 && index_set < n-1 # try all pairs with the smallest before quitting
            # We didn't swap anything in this (large) set, so we try the next set
            index_set+=1
            index_el = 1
        else
            # nothing changed, so we can as well stop
            break
        end
    end
    return sets,set_weights,[weights[s] for s in sets]
end

struct ThreadingInfo
    j_order::Vector{Int} # the order of the clusters for threading
    jl_order::Vector{Tuple{Int,Int}} # the order of the (j,l) blocks for threading
end

function ThreadingInfo(sdp::ClusteredLowRankSDP,n=Threads.nthreads())
    # First we determine the blocksizes of Y. This determines the distribution over the threads
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
    Y_blocksizes = [[size(sdp.A[j][l],1)*subblocksizes[j][l] for l in eachindex(sdp.A[j])] for j in eachindex(sdp.A)]

    # Now we find a good order for the (j,l) pairs:
    # the only relevant parameter here is the blocksize Y_blocksizes[j][l]
    # Computing the step length takes a cholesky ~n^3 . Eigenvalues should also take ~n^3
    jl_order = [(j,l) for j in eachindex(sdp.A) for l in eachindex(sdp.A[j])]
    jl_pair_weights = [Y_blocksizes[j][l]^3 for (j,l) in jl_order]
    setsjl, weightsjl, weight_distjl = distribute_weights_swapping(jl_pair_weights, n)
    # to get these sets on the cores, we need to put the longer sets first
    sort!(setsjl, by=length, rev=true)
    jl_order = jl_order[vcat(setsjl...)]

    # Determine a good order for the clusters j:
    # We base the weights for the clusters on the Cholesky decomposition.
    # In principle we could instead use the time for the solving (P[j]^2*N). Not sure what is better, maybe P[j]^2*N/2 + 1/3*P[j]^3
    j_order_weights = [size(sdp.c[j],1)^3 for j in eachindex(sdp.c)] # [1//2 * size(sdp.c[j],1)^2*size(sdp.b,1) + 1//3 * size(sdp.c[j].1)^3 for j in eachindex(sdp.c)]
    setsj, weightsj, weight_distj = distribute_weights_swapping(j_order_weights, n)
    sort!(setsj, by=length, rev=true)
    j_order = vcat(setsj...)
    return ThreadingInfo(j_order,jl_order)
end
