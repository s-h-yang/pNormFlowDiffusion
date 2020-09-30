# This Julia code is adopted from MATLAB script run_diffusion.m obtained at
# https://github.com/RaniaSalama/Nonlinear_Diffusion/tree/master/LFR_Figures/.
# It implements a nonlinear diffusion method that iteratively applies power
# transformation to node values.

using SparseArrays

function nonlinear_power_diffusion(L::SparseMatrixCSC{Float64,Int},
            Dinv::SparseMatrixCSC{Float64,Int}, seed::Int, h::Real;
            power::Real=0.5, max_iters::Int=100)

    n = size(L, 2)
    u = zeros(Float64, n)
    u[seed] = 1
    for k = 1:max_iters
        u -= h*(L*(u.^power))
        u[findall(x -> x < 0.0, u)] .= 0.0
        u[findall(x -> x > 1.0, u)] .= 1.0
    end
    return Dinv*u
end
