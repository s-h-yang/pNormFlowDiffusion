using DelimitedFiles, SparseArrays

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
        @warn "The graph contains nodes with degree 0."
    end
    return AdjacencyList(adjlist, degree, nv)
end

function adjacency_matrix(G::AdjacencyList)
    n = G.nv
    colptr = Vector{Int}(undef, n+1)
    colptr[1] = 1
    for i = 1:n
        colptr[i+1] = colptr[i] + G.degree[i]
    end
    rowval = Int[]
    for i in 1:n
        for j in G.adjlist[i]
            push!(rowval, j)
        end
    end
    nzval = ones(Int, sum(G.degree))
    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end


function compute_symmetric_laplacian(A::SparseMatrixCSC{Int,Int}, d::Vector{Int})
    ind = [i == 0 ? 0.0 : 1.0 for i in d]
    dsqn = [i == 0 ? 0.0 : 1.0/i for i in sqrt.(d)]
    Dsqn = spdiagm(0 => dsqn)
    return spdiagm(0 => ind) - Dsqn*(A*Dsqn)
end

function compute_normalized_laplacian(A::SparseMatrixCSC{Int,Int}, d::Vector{Int})
    ind = [i == 0 ? 0.0 : 1.0 for i in d]
    dinv = [i == 0 ? 0.0 : 1.0/i for i in d]
    Dinv = spdiagm(0 => dinv)
    return spdiagm(0 => ind) - A*Dinv
end

function sweepcut(G::AdjacencyList, x::Vector{Float64})
    vol_G = sum(G.degree)
    vol_C = 0
    cut_C = 0
    C = Int[]
    best_cluster = Int[]
    best_conductance = 1.0
    x_nzind = findall(!iszero, x)
    x_nzval = x[x_nzind]
    sorted_ind = x_nzind[sortperm(x_nzval, rev=true)]
    for v in sorted_ind
        push!(C, v)
        vol_C += G.degree[v]
        for u in G.adjlist[v]
            if u in C
                cut_C -= 1
            else
                cut_C += 1
            end
        end
        cond_C = cut_C / min(vol_C, vol_G - vol_C)
        if cond_C <= best_conductance
            best_cluster = copy(C)
            best_conductance = cond_C
        end
    end
    return best_cluster, best_conductance
end

function compute_f1(cluster::Vector{Int}, target_cluster::Vector{Int})
    tp = length(intersect(Set(target_cluster),Set(cluster)))
    pr = tp/length(cluster)
    re = tp/length(target_cluster)
    if pr == 0 && re == 0
        f1 = 0
    else
        f1 = 2*pr*re/(pr+re)
    end
    return pr, re, f1
end

function compute_conductance(G::AdjacencyList, cluster::Vector{Int})
    vol_G = sum(G.degree)
    vol_C = sum(G.degree[cluster])
    cut_C = vol_C
    for v in cluster
        for u in G.adjlist[v]
            if u in cluster
                cut_C -= 1
            end
        end
    end
    return cut_C / min(vol_C, vol_G - vol_C)
end
