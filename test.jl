using LightGraphs

include("pNormDiffusion.jl")

# Create a graph from stochastic block model with 10 blocks, each block has
# 100 nodes, internal probability 0.5, external probability 0.01
sbm = StochasticBlockModel(0.5, 0.01, 100, 10)
simple_g = SimpleGraph(1000, 30000, sbm)
G = AdjacencyList(simple_g.fadjlist, degree(simple_g), 1000)

# The total amount of initial mass should be at least two times the volume of
# target cluster. Here we set seed mass to be roughly three times the volume of
# target cluster.
seed_node = 1
seed_mass = 0.3*sum(G.degree)
seedset = Dict(seed_node => seed_mass)

# run p-norm flow diffusion
x = pnormdiffusion(G, seedset, p=4)

# obtain a cluster from x
cluster, conductance = sweepcut(G, x)
println("conductance is ", conductance)

# compute F1 score
tp = length(intersect(Set(1:100),Set(cluster)))
pr = tp/length(cluster)
re = tp/100
f1 = 2*pr*re/(pr+re)
println("F1 socre is ", f1)
