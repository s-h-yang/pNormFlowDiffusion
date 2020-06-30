using SparseArrays, Statistics, Printf

include("pnormdiffusion_tuning.jl")
include("optimal_pagerank.jl")
include("nonlinear.jl")

function run_Sfld_experiment()

    G = read_edgelist("../datasets/Sfld_edgelist.txt")
    A = adjacency_matrix(G)
    Ls = compute_symmetric_laplacian(A, G.degree)
    Ln = compute_normalized_laplacian(A, G.degree)
    dinv = [1.0/i for i in G.degree]
    Dinv = spdiagm(0 => dinv)
    n = G.nv

    communities = open("../datasets/Sfld_community.txt", "r") do f
        [parse.(Int, split(line)) for line in eachline(f)]
    end
    community_num = length(communities)
    method_num = 4
    conds_mean = zeros(community_num, method_num)
    f1s_mean = zeros(community_num, method_num)

    for (c,target_cluster) in enumerate(communities)

        conds = zeros(length(target_cluster), method_num)
        f1s = zeros(length(target_cluster), method_num)
        trial_num = length(target_cluster)

        for (i,seed_node) in enumerate(target_cluster)

            @printf("cluster %d of %d, seed node %d of %d\n", c, community_num,
                    i, trial_num)

            _, _, conds[i,1], f1s[i,1] = pnormdiffusion_tuning(G, seed_node,
                    target_cluster, collect(1:10), p=2, max_iters=100)

            _, _, conds[i,2], f1s[i,2] = pnormdiffusion_tuning(G, seed_node,
                    target_cluster, collect(1:10), p=4, max_iters=100,
                    cm_tol=1.0e-6)

            _, _, conds[i,3], f1s[i,3] = optimal_pagerank(G, seed_node,
                    target_cluster, Ls, collect(1:10))

            u = nonlinear_power_diffusion(Ln, Dinv, seed_node, 0.001, power=0.5)
            cluster, conds[i,4] = sweepcut(G, u)
            _, _, f1s[i,4] = compute_f1(cluster, target_cluster)

        end

        conds_mean[c,:] = mean(conds, dims=1)
        f1s_mean[c,:] = mean(f1s, dims=1)

    end

    return conds_mean, f1s_mean

end

conds_mean, f1s_mean = run_Sfld_experiment()

open("Sfld_conductances.txt", "w") do f
    @printf(f, "%s\t%s\t%s\t%s\n", "p=2", "p=4", "appr", "nonlinear")
    for m in eachrow(conds_mean)
        @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", m[1], m[2], m[3], m[4])
    end
end

open("Sfld_F1scores.txt", "w") do f
    @printf(f, "%s\t%s\t%s\t%s\n", "p=2", "p=4", "appr", "nonlinear")
    for m in eachrow(f1s_mean)
        @printf(f, "%.4f\t%.4f\t%.4f\t%.4f\n", m[1], m[2], m[3], m[4])
    end
end
