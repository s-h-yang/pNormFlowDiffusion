To produce Figure 2 (resp. Figure 5) in the conference version (resp. arxiv version), simply `include("LFR_experiment.jl")`. 
It requires [PyCall](https://github.com/JuliaPy/PyCall.jl) and [PyPlot](https://github.com/JuliaPy/PyPlot.jl) to generate the plots.

To re-compute the numbers in Table 1 (resp. Table 3) in the conference version (resp. arxiv version)
- `include("Facebook_experiments.jl")` for the two Facebook social networks;
- `include("Sfld_experiment.jl")` for the biological network;
- follow the instructions in Orkut_experiment.jl and then `include("Orkut_experiment.jl")`.
