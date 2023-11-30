casename = "case118_3" 
n_areas = 3
need_csv = 0
pq_lower = 150
pq_upper = 800
vt_lower = 3000
vt_upper = 5000
T = 1
run_num = 1

if lastindex(ARGS) >= 9
    casename = ARGS[1]
    n_areas = parse(Int,ARGS[2])
    need_csv = parse(Int, ARGS[3])
    pq_lower = parse(Int, ARGS[4])
    pq_upper = parse(Int, ARGS[5])
    vt_lower = parse(Int, ARGS[6])
    vt_upper = parse(Int, ARGS[7])
    T = parse(Int, ARGS[8])
    run_num = parse(Int, ARGS[9])
else
    println("Using default, no command line arguments given")
end

using Distributed 
addprocs(n_areas)

@everywhere using Pkg 
@everywhere Pkg.activate(".")

using LinearAlgebra
@everywhere using PowerModelsADA 
using JuMP 
@everywhere using Ipopt 
using BSON 
include("linucb_functions.jl")

case_path = "data/$casename.m"
data = parse_file(case_path)
if (need_csv == 1)
    partition_path= "data/$casename"*"_$n_areas.csv"
    assign_area!(data, partition_path)
end

model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1000
optimizer = Ipopt.Optimizer 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

pq_bounds = [pq_lower,pq_upper]
vt_bounds = [vt_lower,vt_upper]

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)
lambda = 0.05 

initial_iters = 20

reward,alpha_config,trace_params = run_linucb(T,data_area,pq_bounds,vt_bounds,initial_config,optimizer,lambda,initial_iters)
bson("data/hyperband/linucb_$casename"*"_$pq_lower"*"_$pq_upper"*"_$vt_lower"*"_$vt_upper"*"_$run_num.jl", Dict("alpha_config" => alpha_config, "reward" => reward, "trace" => trace_params))
