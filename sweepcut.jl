function sweepcut(adjlist::Vector{Vector{Int}}, degree::Vector{Int}, x::Vector{Float64})

    vol_G = sum(degree)
    vol_C = 0
    cut_C = 0
    C = Int[]
    best_cluster = Int[]
    best_conductance = 1.0
    sorted_x = reverse!(sortperm(x))

    for v in sorted_x[1:count(!iszero,x)]
        push!(C, v)
        vol_C += degree[v]
        for u in adjlist[v]
            if u in C
                cut_C -= 1
            else
                cut_C += 1
            end
        end
        cond_C = cut_C / min(vol_C, vol_G-vol_C)
        if cond_C <= best_conductance
            best_cluster = copy(C)
            best_conductance = cond_C
        end
    end

    return best_cluster, best_conductance
end
