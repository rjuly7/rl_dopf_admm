casename = "case118_3"
n_areas = 3
need_csv = 0
budget = 0.1
tidx = 4 
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

using Distributed 
if nprocs() == 1
    addprocs(3)
end
@everywhere using Pkg 
@everywhere Pkg.activate(".")
@everywhere using LinearAlgebra
@everywhere using PowerModelsADA 
@everywhere using PowerModels
@everywhere using JuMP 
@everywhere using Ipopt 
using BSON 
using Random 
Random.seed!(123)
@everywhere include("linucb_functions.jl")

case_path = "data/$casename.m"
data = parse_file(case_path)
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1200
optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

pq_bounds = [pq_lower,pq_upper]
vt_bounds = [vt_lower,vt_upper]

lambda = 0.1

initial_config = set_hyperparameter_configuration(data_area,400,4000)

linucb_agents = run_linucb(T,data_area,data,pq_bounds,vt_bounds,optimizer,lambda,initial_config)
bson("data/hyperband/linucb_$run_num.jl", Dict("linucb_agents" => linucb_agents))

# region_bounds=[(0.01,10),(0.001,1),(1e-4,0.1)]
# linucb_agents = Dict{Int,Any}()
# for n in eachindex(region_bounds)
#     linucb_agents[n] = initialize_lin_ucb(pq_bounds, vt_bounds, region_bounds, data_area, lambda)
# end
# rr = run_then_return_val_loss_sp(data_area,linucb_agents,optimizer)