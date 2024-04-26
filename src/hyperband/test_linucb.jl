casename = "case141_added_gens_4" 
pq_lower = 150
pq_upper = 800
vt_lower = 3000
vt_upper = 5000
T = 20
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
include("linucb_functions.jl")
include("test_linucb_functions.jl")

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

configs = get_alpha_configs(linucb_agents)
rewards = run_with_configs(data_area,configs,optimizer)

plot(linucb_agents[1]["trace_params"]["reward"])