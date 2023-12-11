using Pkg 
Pkg.activate(".")
using Plots 
using BSON 

casename = "case118_3"
n_areas = 3
need_csv = 0
run_num = "1"

case_path = "data/$casename.m"
data = parse_file(case_path)
if (need_csv == 1)
    partition_path= "data/$casename"*"_$n_areas.csv"
    assign_area!(data, partition_path)
end

if casename == "case118_3"
    pqs_l = [100, 100, 100, 100, 100, 150, 200, 250, 300, 350, 375]
    pqs_u = [4000, 3600, 3200, 2800, 2000, 1200, 900, 700, 600, 500, 450]

    vts_l = [800, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 3600, 3800, 3900]
    vts_u = [9000, 7200, 7000, 6500, 6000, 5600, 5200, 4800, 4400, 4200, 4100]
elseif casename == "case30"
#case30
    pqs_l = [100, 100, 150, 200, 250, 300, 350]
    pqs_u = [2800, 2000, 1200, 900, 700, 600, 500]

    vts_l = [1600, 2000, 2400, 2800, 3200, 3600, 3800]
    vts_u = [6500, 6000, 5600, 5200, 4800, 4400, 4200]
end

stuff = BSON.load("data/hyperband/$casename"*"_results.jl")
n_iters_all = stuff["n_iters_all"]
n_iters_mean = [mean(n_iters_all[i]) for i in eachindex(n_iters_all)]

model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1000
optimizer = Ipopt.Optimizer 

initial_iters = 20 
alpha_pq = 400
alpha_vt = 4000 
data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)
data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)
baseline_r = run_then_return_val_loss(deepcopy(data_area),initial_config,initial_config,optimizer,initial_iters)
baseline_iter = 200 - baseline_r 
spread = pqs_u - pqs_l + vts_u - vts_l 
plot(spread,n_iters_mean,ylim=(0,baseline_iter + 10),linewidth=2,label="LinUCB",xlabel="Length of search space",ylabel="Iterations to converge")
plot!(spread,baseline_iter*ones(length(spread)),linewidth=2,label="Baseline")
savefig("data/figs/$casename"*"_iters_vs_space.png")

plot(spread, pqs_l, fillrange = pqs_u, fillalpha = 0.35, label = "PQ values",xlabel="Length of search space", ylabel="Search values")
plot!(spread, vts_l, fillrange = vts_u, fillalpha = 0.35, label = "VT values")
plot!(spread,4000*ones(length(spread)),line=:dash,linewidth=2,color="black",label="Baseline PQ")
plot!(spread,400*ones(length(spread)),line=:dot,linewidth=2,color="black",label="Baseline VT")
savefig("data/figs/$casename"*"_bounds.png")