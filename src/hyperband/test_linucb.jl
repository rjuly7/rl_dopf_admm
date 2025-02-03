casename = "case118_3"
pq_lower = 150
pq_upper = 800
vt_lower = 3000
vt_upper = 5000
T = 10
run_num = 1

if lastindex(ARGS) >= 7
    casename = ARGS[1]
    pq_lower = parse(Int, ARGS[2])
    pq_upper = parse(Int, ARGS[3])
    vt_lower = parse(Int, ARGS[4])
    vt_upper = parse(Int, ARGS[5])
    T = parse(Int, ARGS[6])
    run_num = parse(Int, ARGS[7])
else
    println("Using default, no command line arguments given")
end
path = pwd()
using Pkg 
Pkg.activate(".")

using Distributed 
using LinearAlgebra
using PowerModelsADA 
using PowerModels
using JuMP 
using Ipopt 
using BSON 
using Random 
using Plots 
Random.seed!(123)
include("$path/src/hyperband/linucb_functions.jl")
include("$path/src/hyperband/test_linucb_functions.jl")

case_path = "data/$casename.m"
data = parse_file(case_path)
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1200
optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

linucb_agents = BSON.load("data/hyperband/linucb_$run_num.jl")["linucb_agents"]
initial_config = set_hyperparameter_configuration(data_area,400,4000)
configs = get_alpha_configs(linucb_agents)
N = 3
# for i=1:N 
#     configs[i] = initial_config
# end
reached_bound_iter,iteration = run_with_configs(deepcopy(data_area),configs,optimizer,linucb_agents,N)
# println(iteration)

# alpha_pq = 400
# alpha_vt = 4000 
# initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)
# configs_f = Dict(i => initial_config for i in keys(configs))
# rewards_f, iteration_f = run_with_configs(deepcopy(data_area),configs_f,optimizer)

