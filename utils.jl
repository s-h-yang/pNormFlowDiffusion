using DelimitedFiles

include("struct.jl")

function read_edgelist(source; delim::AbstractChar='\t')
    edgelist = readdlm(source, delim, Int, '\n')
    if minimum(edgelist) != 1
        error("Node index must start from 1.")
    end
    nv = maximum(edgelist)
    adjlist = [Int[] for i in 1:nv]
    for edge in eachrow(edgelist)
        push!(adjlist[edge[1]], edge[2])
        push!(adjlist[edge[2]], edge[1])
    end
    degree = [length(l) for l in adjlist]
    if any(d -> d == 0, degree)
        error("Some nodes have degree 0.")
    end
    return AdjacencyList(adjlist, degree, nv)
end

function sweepcut(G::AdjacencyList, x::Vector{Float64})
    vol_G = sum(G.degree)
    vol_C = 0
    cut_C = 0
    C = Int[]
    best_cluster = Int[]
    best_conductance = 1.0
    sorted_x = reverse!(sortperm(x))
    for v in sorted_x[1:count(!iszero,x)]
        push!(C, v)
        vol_C += G.degree[v]
        for u in G.adjlist[v]
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
