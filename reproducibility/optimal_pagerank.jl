using LinearAlgebra
using Arpack

include("../struct.jl")
include("../utils.jl")
include("pagerank.jl")

function optimal_pagerank(G::AdjacencyList, seed::Int, target::Vector{Int},
            L::SparseMatrixCSC{Float64,Int}, vscales; tol::Float64=1.0e-7,
            max_iters::Int=1000)

    n = G.nv
    s = zeros(Float64, n)
    s[seed] = 1/sqrt(G.degree[seed])
    L_sub = @view L[:,target]
    L_sub = L_sub[target, :]
    lambda = eigs(L_sub; nev=1, which=:SM)[1][1]
    target_volume = sum(G.degree[target])
    alphas = [lambda/8, lambda/4, lambda/2, lambda, 2*lambda]
    alphas = alphas[findall(x -> x < 1, alphas)]
    rhos = 1 ./ (vscales*target_volume)

    best_p = []
    best_cluster = []
    best_f1 = 0.0
    best_cond = 1.0
    for alpha in alphas
        for rho in rhos
            p = pagerank(G, s, rho, alpha, tol=tol, max_iters=max_iters)
            x = p ./ G.degree
            ind = findall(iszero, G.degree)
            x[ind] .= 0.0
            cluster, cond = sweepcut(G, x)
            if cond <= best_cond
                best_p = copy(p)
                best_cluster = copy(cluster)
                best_cond = cond
                _, _, best_f1 = compute_f1(cluster, target)
            end
        end
    end
    return best_p, best_cluster, best_cond, best_f1
end
