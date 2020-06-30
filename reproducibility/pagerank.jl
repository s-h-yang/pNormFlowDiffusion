# Implements the L1-regularized PageRank. See
# "Variational Perspective on Local Graph Clustering. K. Fountoulakis et al."
using SparseArrays

include("../struct.jl")

function pagerank(G::AdjacencyList, s::Array{Float64,1}, rho::Float64,
            alpha::Float64; tol::Float64=1.0e-7, max_iters::Int=1000)

    d = G.degree
    g = -alpha*s
    p = zeros(Float64, G.nv)
    S = Set(findall(!iszero, s))
    for k = 1:max_iters
        err = 0.0
        C = [v for v in S if -g[v] > rho*alpha*sqrt(d[v]) + tol]
        for v in shuffle!(C)
            prev = p[v]
            p[v] -= g[v]
            p[v] = max(p[v] - rho*alpha*sqrt(d[v]), 0)
            push = p[v] - prev
            if push > 0
                err = max(err, push)
                g[v] += (alpha + (1 - alpha)/2)*push
                for u in G.adjlist[v]
                    if g[u] == 0
                        push!(S, u)
                    end
                    g[u] -= (1 - alpha)/(2*sqrt(d[u])*sqrt(d[v]))*push
                end
            end
        end
        if err < tol
            break
        end
    end
    return p
end
