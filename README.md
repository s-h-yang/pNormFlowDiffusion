# p-Norm Flow Diffusion

This repository contains the code to solve primal and dual p-Norm Flow Diffusion problems in the paper "p-Norm Flow Diffusion for Local Graph clustering. Kimon Fountoulakis, Di Wang, Shenghao Yang." The code returns dual variable values that provide node embeddings. The primal flow values are easily recovered from primal-dual optimality conditions. For details see [paper](https://arxiv.org/abs/2005.09810), [slides](http://www1.icsi.berkeley.edu/~kfount/pdf/siammds20_pnorm), and [video](https://www.youtube.com/watch?v=X6V11ZFCkk8&feature=emb_title). To reproduce the results in the paper, use the scripts in [reproducibility](https://github.com/s-h-yang/pNormFlowDiffusion/tree/master/reproducibility).

# Example

Below is a simple demonstration from test.jl on how to use p-Norm Flow Diffusion for local clustering on a graph sampled from stochastic block model.
```julia
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

# Run p-Norm Flow Diffusion
x = pnormdiffusion(G, seedset, p=4)

# Obtain a cluster by applying sweepcut on x
cluster, conductance = sweepcut(G, x)
println("conductance is ", conductance)

# Compute F1 score
tp = length(intersect(Set(1:100),Set(cluster)))
pr = tp/length(cluster)
re = tp/100
f1 = 2*pr*re/(pr+re)
println("F1 socre is ", f1)
```

More examples with visualizations are in the [demo](https://github.com/s-h-yang/pNormFlowDiffusion/tree/master/demo) folder.
