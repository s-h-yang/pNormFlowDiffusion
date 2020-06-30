include("../pNormDiffusion.jl")

function pnormdiffusion_tuning(G::AdjacencyList, seed_node::Int,
            target_cluster::Vector{Int}, vscales; p::Real=2, max_iters=25,
            epsilon=1.0e-3, cm_tol=1.0e-2)

    graph_volume = sum(G.degree)
    target_volume = sum(G.degree[target_cluster])
    best_x = []
    best_cluster = []
    best_cond = 1
    best_f1 = 0

    smax = graph_volume/target_volume
    if any(x -> x > smax, vscales)
        filter!(x -> x < smax, vscales)
        vscales = vcat(vscales, smax)
    end

    for s in vscales
        seed_mass = s*target_volume
        if seed_mass > graph_volume
            break
        end
        seedset = Dict(seed_node => seed_mass)
        x = pnormdiffusion(G, seedset, p=p, max_iters=max_iters,
                epsilon=epsilon, cm_tol=cm_tol)
        cluster, cond = sweepcut(G, x)
        _, _, f1 = compute_f1(cluster, target_cluster)
        if cond <= best_cond
            best_x = copy(x)
            best_cluster = copy(cluster)
            best_cond = cond
            best_f1 = f1
        end
    end
    return best_x, best_cluster, best_cond, best_f1
end
