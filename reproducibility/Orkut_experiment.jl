# To run this experiment, download the com-Orkut dataset from
# http://snap.stanford.edu/data/com-Orkut.html and place the edgelist file
# "com-orkut.ungraph.txt" under the path ../datasets/
using SparseArrays, Statistics, Printf

include("../pNormDiffusion.jl")
include("optimal_pagerank.jl")
include("nonlinear.jl")

function run_orkut_experiment(path_to_edgelist, path_to_community_file)

    G = read_edgelist(path_to_edgelist)
    A = adjacency_matrix(G)
    L = compute_symmetric_laplacian(A, G.degree)
    n = G.nv

    communities = open(path_to_community_file, "r") do f
        [parse.(Int, split(line)) for line in eachline(f)]
    end
    community_num = length(communities)
    method_num = 3
    conds_mean = zeros(community_num, method_num)
    f1s_mean = zeros(community_num, method_num)

    for (c,target_cluster) in enumerate(communities)

        target_volume = sum(G.degree[target_cluster])
        seed_mass = min(5*target_volume, sum(G.degree))

        conds = zeros(length(target_cluster), method_num)
        f1s = zeros(length(target_cluster), method_num)
        trial_num = length(target_cluster)

        for (i,seed_node) in enumerate(target_cluster)

            @printf("cluster %d of %d, seed node %d of %d\n", c, community_num,
                    i, trial_num)

            seedset = Dict(seed_node => seed_mass)

            x = pnormdiffusion(G, seedset, p=2, max_iters=200)
            cluster, conds[i,1] = sweepcut(G, x)
            _, _, f1s[i,1] = compute_f1(cluster, target_cluster)

            x = pnormdiffusion(G, seedset, p=4, max_iters=200)
            cluster, conds[i,2] = sweepcut(G, x)
            _, _, f1s[i,2] = compute_f1(cluster, target_cluster)

            _, _, conds[i,3], f1s[i,3] = optimal_pagerank(G, seed_node,
                    target_cluster, L, 5, max_iters=200)

        end

        conds_mean[c,:] = mean(conds, dims=1)
        f1s_mean[c,:] = mean(f1s, dims=1)

    end

    return conds_mean, f1s_mean

end

path1 = "../datasets/com-orkut.ungraph.txt"
path2 = "../datasets/Orkut_community.txt"
conds_mean, f1s_mean = run_orkut_experiment(path1, path2)
open("Orkut_conductances.txt", "w") do f
    @printf(f, "%s\t%s\t%s\n", "p=2", "p=4", "appr")
    for m in eachrow(conds_mean)
        @printf(f, "%.4f\t%.4f\t%.4f\n", m[1], m[2], m[3])
    end
end
open("Orkut_F1scores.txt", "w") do f
    @printf(f, "%s\t%s\t%s\n", "p=2", "p=4", "appr")
    for m in eachrow(f1s_mean)
        @printf(f, "%.4f\t%.4f\t%.4f\n", m[1], m[2], m[3])
    end
end
