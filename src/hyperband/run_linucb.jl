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
Random.seed!(123)
include("linucb_functions.jl")

run_num = 1 

case_path = "data/case118_3.m"
data = parse_file(case_path)
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1000
optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

pq_bounds = [200,800]
vt_bounds = [3000,5000]

lambda = 0.1

T = 2
linucb_agents = run_linucb(T,data_area,data,pq_bounds,vt_bounds,optimizer,lambda)
bson("data/hyperband/linucb_$run_num.jl", Dict("linucb_agents" => linucb_agents))

# region_bounds=[(0.01,10),(0.001,1),(1e-4,0.1)]
# linucb_agents = Dict{Int,Any}()
# for n in eachindex(region_bounds)
#     linucb_agents[n] = initialize_lin_ucb(pq_bounds, vt_bounds, region_bounds, data_area, lambda)
# end
# rr = run_then_return_val_loss_sp(data_area,linucb_agents,optimizer)