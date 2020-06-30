# The LFR synthetic datasets used for this experiment are obtained from
# https://github.com/RaniaSalama/Nonlinear_Diffusion.
# The parameters for the LFR model are:
# n = 1000
# average degree = 10
# maximum degree = 50
# minimum community size = 20
# maximum community size = 100
# power law exponent for the degree distribution = 2
# power law exponent for the community size distribution = 1
using SparseArrays, Printf, Statistics
using PyCall, PyPlot

include("../pNormDiffusion.jl")
include("optimal_pagerank.jl")
include("nonlinear.jl")

fmt(mu::Float64) = @sprintf("%.2f", mu)

function run_LFR_experiments()

    graph_num = 16
    method_num = 5
    trial_num = 100

    conds_mean = zeros(graph_num, method_num)
    conds_std = zeros(graph_num, method_num)
    f1s_mean = zeros(graph_num, method_num)
    f1s_std = zeros(graph_num, method_num)

    g_cnt = 1

    for mu in 0.1:0.02:0.4

        G = read_edgelist("../datasets/LFR/edgelist_mu="*fmt(mu)*".txt")
        A = adjacency_matrix(G)
        Ls = compute_symmetric_laplacian(A, G.degree)
        Ln = compute_normalized_laplacian(A, G.degree)
        dinv = [1.0/i for i in G.degree]
        Dinv = spdiagm(0 => dinv)
        n = G.nv

        communities = open("../datasets/LFR/community_mu="*fmt(mu)*".txt", "r") do f
            [parse.(Int, split(line)) for line in eachline(f)]
        end
        node_membership = Dict{Int,Int}(i => 0 for i in 1:n)
        for (c, community) in enumerate(communities)
            for i in community
                node_membership[i] = c
            end
        end

        seeds = rand(1:n, trial_num)
        conds = zeros(trial_num, method_num)
        f1s = zeros(trial_num, method_num)

        for (t,seed_node) in enumerate(seeds)

            println("mu = "*fmt(mu)*", trial ", t)

            target_cluster = communities[node_membership[seed_node]]
            target_volume = sum(G.degree[target_cluster])
            seed_mass = min(5*target_volume, sum(G.degree))
            seedset = Dict(seed_node => seed_mass)

            x = pnormdiffusion(G, seedset, p=2, max_iters=50)
            cluster, conds[t,1] = sweepcut(G, x)
            _, _, f1s[t,1] = compute_f1(cluster, target_cluster)

            x = pnormdiffusion(G, seedset, p=4, max_iters=100)
            cluster, conds[t,2] = sweepcut(G, x)
            _, _, f1s[t,2] = compute_f1(cluster, target_cluster,)

            x = pnormdiffusion(G, seedset, p=8, max_iters=200)
            cluster, conds[t,3] = sweepcut(G, x)
            _, _, f1s[t,3] = compute_f1(cluster, target_cluster)

            _, _, conds[t,4], f1s[t,4] = optimal_pagerank(G, seed_node,
                                            target_cluster, Ls, 5)

            u = nonlinear_power_diffusion(Ln, Dinv, seed_node, 0.001, power=0.5)
            cluster, conds[t,5] = sweepcut(G, u)
            _, _, f1s[t,5] = compute_f1(cluster, target_cluster)

        end

        conds_mean[g_cnt,:] = mean(conds, dims=1)
        conds_std[g_cnt,:] = std(conds, dims=1)
        f1s_mean[g_cnt,:] = mean(f1s, dims=1)
        f1s_std[g_cnt,:] = std(f1s, dims=1)

        g_cnt += 1

    end

    return conds_mean, conds_std, f1s_mean, f1s_std

end

conds_mean, conds_std, f1s_mean, f1s_std = run_LFR_experiments()

plt = pyimport("matplotlib.pyplot")
mu = collect(0.1:0.02:0.4)
plt.figure(figsize=(6,5))
plt.plot(mu, conds_mean[:,1], label="p = 2", color="red")
plt.plot(mu, conds_mean[:,2], label="p = 4", color="tab:green")
plt.plot(mu, conds_mean[:,3], label="p = 8", color="blue")
plt.plot(mu, conds_mean[:,4], label="pagerank", color="cyan")
plt.plot(mu, conds_mean[:,5], label="nonlinear power", color="magenta")
plt.fill_between(mu, conds_mean[:,1]-conds_std[:,1].^2,
        conds_mean[:,1]+conds_std[:,1].^2, alpha=0.5, color="red")
plt.fill_between(mu, conds_mean[:,2]-conds_std[:,2].^2,
        conds_mean[:,2]+conds_std[:,2].^2, alpha=0.5, color="tab:green")
plt.fill_between(mu, conds_mean[:,3]-conds_std[:,3].^2,
        conds_mean[:,3]+conds_std[:,3].^2, alpha=0.5, color="blue")
plt.fill_between(mu, conds_mean[:,4]-conds_std[:,4].^2,
        conds_mean[:,4]+conds_std[:,4].^2, alpha=0.5, color="cyan")
plt.fill_between(mu, conds_mean[:,5]-conds_std[:,5].^2,
        conds_mean[:,5]+conds_std[:,5].^2, alpha=0.5, color="magenta")
plt.xlabel("mu", size=20)
plt.ylabel("Conductance", size=20)
plt.ylim((0.1,0.6))
plt.xticks((0.1, 0.2, 0.3, 0.4), size=18)
plt.yticks(size=18)
plt.legend(fontsize=13, loc="lower right")
plt.savefig("LFR_conductance.pdf", bbox_inches="tight", format="pdf")


plt.figure(figsize=(6,5))
plt.plot(mu, f1s_mean[:,1], label="p = 2", color="red")
plt.plot(mu, f1s_mean[:,2], label="p = 4", color="tab:green")
plt.plot(mu, f1s_mean[:,3], label="p = 8", color="blue")
plt.plot(mu, f1s_mean[:,4], label="pagerank", color="cyan")
plt.plot(mu, f1s_mean[:,5], label="nonlinear power", color="magenta")
plt.fill_between(mu, f1s_mean[:,1]-f1s_std[:,1].^2,
        f1s_mean[:,1]+f1s_std[:,1].^2, alpha=0.5, color="red")
plt.fill_between(mu, f1s_mean[:,2]-f1s_std[:,2].^2,
        f1s_mean[:,2]+f1s_std[:,2].^2, alpha=0.5, color="tab:green")
plt.fill_between(mu, f1s_mean[:,3]-f1s_std[:,3].^2,
        f1s_mean[:,3]+f1s_std[:,3].^2, alpha=0.5, color="blue")
plt.fill_between(mu, f1s_mean[:,4]-f1s_std[:,4].^2,
        f1s_mean[:,4]+f1s_std[:,4].^2, alpha=0.5, color="cyan")
plt.fill_between(mu, f1s_mean[:,5]-f1s_std[:,5].^2,
        f1s_mean[:,5]+f1s_std[:,5].^2, alpha=0.5, color="magenta")
plt.xlabel("mu", size=20)
plt.ylabel("F1 measure", size=20)
plt.ylim((0.3,1))
plt.xticks((0.1, 0.2, 0.3, 0.4), size=18)
plt.yticks(size=18)
plt.legend(fontsize=13, loc="lower left")
plt.xticks((0.1, 0.2, 0.3, 0.4))
plt.savefig("LFR_f1.pdf", bbox_inches="tight", format="pdf")
