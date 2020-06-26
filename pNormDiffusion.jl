using Random

include("struct.jl")
include("utils.jl")

"""
Computes node embeddings from p-norm flow diffusion on an undirected unweighted
graph G. Extentions to weighted graphs are straightforward.

Inputs:
            G - Adjacency list representation of graph.
                Node indices must start from 1 and end with n, where n is the
                number of nodes in G.

      seedset - Dictionary specifying seed node(s) and seed mass as
                (key, value) pairs.

            p - Specifies the p-norm in primal p-norm flow objective.

           mu - Smoothing parameter that smoothes the p-norm objective.
                Setting mu=0 (default) suffices for almost all practical
                purposes.

    max_iters - Maximum number of passes for Random Permutation Coordinate
                Minimization. A single pass goes over all nodes that violate
                KKT conditions.

      epsilon - Tolerance on the maximum excess mass on nodes (which is
                equivalent to the maximum primal infeasibility).
                Diffusion process terminates whenever the excess mass is no
                greater than epsilon on all nodes.

       cm_tol - Tolerance in the approximate coordinate minimization step
                (required only for p > 2).
                Approximate coordinate minimization in this diffusion setting
                is equivalent to performing inexact line-seach on coordinate
                descent stepsizes.

Returns:
            x - Node embeddings.
                Applying sweepcut on x produces the final output cluster.
"""

function pnormdiffusion(G::AdjacencyList, seedset::Dict{Int,T} where T<:Real;
            p::Real, mu::Real=0, max_iters::Int=50, epsilon::Float64=1.0e-3,
            cm_tol::Float64=1.0e-2)

    mass = zeros(Float64, G.nv)
    for (v,m) in seedset
        mass[v] = Float64(m)
    end
    x = zeros(Float64, G.nv)

    if p == 2
        l2opt!(x, mass, G.adjlist, G.degree, max_iters, epsilon)
    elseif p > 2
        lpopt!(x, mass, G.adjlist, G.degree, max_iters, epsilon, p, mu, cm_tol)
    else
        error("p should be >= 2.")
    end

    return x
end

function l2opt!(x::Vector{Float64}, mass::Vector{Float64},
            adjlist::Vector{Vector{Int}}, degree::Vector{Int}, max_iters::Int,
            epsilon::Float64)

    S = findall(!iszero, mass)
    for _ in 1:max_iters
        T = [v for v in S if mass[v] > degree[v] + epsilon]
        if isempty(T)
            break
        end
        for v in shuffle!(T)
            push = (mass[v] - degree[v])/degree[v]
            x[v] += push
            mass[v] = degree[v]
            for u in adjlist[v]
                if mass[u] == 0
                    push!(S, u)
                end
                mass[u] += push
            end
        end
    end
end

function lpopt!(x::Vector{Float64}, mass::Vector{Float64},
            adjlist::Vector{Vector{Int}}, degree::Vector{Int}, max_iters::Int,
            epsilon::Float64, p::Real, mu::Real, cm_tol::Float64)

    q = p/(p - 1)
    mass0 = copy(mass)
    S = findall(!iszero, mass)
    for _ in 1:max_iters
        T = [v for v in S if mass[v] > degree[v] + epsilon]
        if isempty(T)
            break
        end
        for v in shuffle!(T)
            x_v_prev = x[v]
            push_node_v!(x, mass, v, adjlist, degree, mass0, q, mu, cm_tol)
            for u in adjlist[v]
                if mass[u] == 0
                    push!(S, u)
                end
                update_mass_u!(x, mass, u, v, x_v_prev, q, mu)
            end
        end
    end
end

"""
Pushes out the excess mass on node v to its neighbors.
This is done by simply increasing x[v], the incumbent node embedding for node v.
"""
function push_node_v!(x::Vector{Float64}, mass::Vector{Float64}, v::Int,
            adjlist::Vector{Vector{Int}}, degree::Vector{Int},
            mass0::Vector{Float64}, q::Float64, mu::Real, cm_tol::Float64)
    L = x[v]
    U = L + 1
    while compute_mass_v(x, v, U, adjlist, mass0, q, mu) > degree[v]
        L = U
        U *= 2
    end
    tol = max(cm_tol, 2*eps(U));
    while abs(U - L) > tol
        M = (L + U)/2
        if compute_mass_v(x, v, M, adjlist, mass0, q, mu) > degree[v]
            L = M
        else
            U = M
        end
    end
    x[v] = (L + U)/2
    mass[v] = compute_mass_v(x, v, x[v], adjlist, mass0, q, mu)
end

function update_mass_u!(x::Vector{Float64}, mass::Vector{Float64}, u::Int,
            v::Int, x_v_prev::Float64, q::Float64, mu::Real)
    mass[u] += flow_uv(x[u], x_v_prev, q, mu) - flow_uv(x[u], x[v], q, mu)
end

"""
Computes the net mass on node v (i.e. initial mass plus all incoming flows).
"""
function compute_mass_v(x, v, x_v, adjlist, mass0, q, mu)
    mass_v = mass0[v]
    for u in adjlist[v]
        mass_v += flow_uv(x[u], x_v, q, mu)
    end
    return mass_v
end

"""
Computes the amount of flow from node u to node v, given node embeddings x_u for
node u and x_v for node v.
"""
function flow_uv(x_u, x_v, q, mu)
    return mu > 0 ? ((x_u - x_v)^2 + mu^2)^(q/2 - 1) * (x_u - x_v) :
            abs(x_u - x_v)^(q - 1) * sign(x_u - x_v)
end
