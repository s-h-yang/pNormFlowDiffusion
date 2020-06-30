To produce Figure 2 in the paper, simply `include("LFR_experiment.jl")`. It requires `PyCall` and `PyPlot` to generate the plots.

To re-compute the numbers in Table 1 in the paper
- `include("Facebook_experiments.jl")` for the two Facebook social networks;
- `include("Sfld_experiment.jl")` for the biological network;
- follow the instructions in `Orkut_experiment.jl` and then `include("Orkut_experiment.jl")`.
