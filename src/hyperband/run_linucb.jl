using Pkg 
Pkg.activate(".")

using Distributed 
using LinearAlgebra
using PowerModelsADA 
using JuMP 
using Gurobi 
using Ipopt 
using BSON 
include("linucb_functions.jl")

run_num = 1 

case_path = "data/case118_3.m"
data = parse_file(case_path)
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1000
optimizer = Ipopt.Optimizer 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

pq_bounds = [150,800]
vt_bounds = [3000,5000]

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)
lambda = 0.1

T = 1
initial_iters = 30
reward,alpha_config,trace_params = run_linucb(T,data_area,pq_bounds,vt_bounds,initial_config,optimizer,lambda,initial_iters)
bson("data/hyperband/linucb_$run_num.jl", Dict("alpha_config" => alpha_config, "reward" => reward, "trace" => trace_params))

