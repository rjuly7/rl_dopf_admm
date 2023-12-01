using Distributed 
using Gurobi 
using LinearAlgebra
using Ipopt 
using JuMP 
using BSON 
using Plots 
using PowerModelsADA 
include("linucb_functions.jl")

run_num = 1
casename = "case30"
n_areas = 3
need_csv = 1
stuff = BSON.load("data/hyperband/linucb_$casename"*"_$run_num.jl")
trace_params = stuff["trace"]

hat_thetas = []
for i in eachindex(trace_params["V"])
    V = trace_params["V"][i]
    y = trace_params["y"][i]
    push!(hat_thetas, inv(V)*y)
end

plot(trace_params["reward"][200:end])

hat_theta = hat_thetas[end]
V = trace_params["V"][end]
inv_V = inv(V)

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

pq_bounds = [150,800]
vt_bounds = [3000,5000]

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

lower_bounds,upper_bounds = get_bounds(pq_bounds,vt_bounds,data_area)
alpha_config,alpha_vector = get_hyperparameter_configuration(deepcopy(data_area),pq_bounds,vt_bounds) #pull initial config 
nv = length(alpha_vector)
T = 500 
beta = 1 + sqrt(2*log(T)+nv*log((nv+T)/nv))

model = Model(Ipopt.Optimizer)
#set_optimizer_attribute(model, "NonConvex", 2)
@variable(model, lower_bounds[i] <= a[i=1:nv] <= upper_bounds[i])
#@objective(model, Max, dot(hat_theta,a))
@variable(model, u)
@objective(model, Max, dot(hat_theta,a) + sqrt(beta)*u)
@constraint(model, transpose(a)*inv_V*a == u^2)
optimize!(model)
println(termination_status(model))

alpha_vector = value.(a)
alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
initial_iters = 20
reward = run_then_return_val_loss(deepcopy(data_area),alpha_config,initial_config,optimizer,initial_iters)

println("Iterations to converge: ", 200 - reward)

model = Model(Ipopt.Optimizer)
#set_optimizer_attribute(model, "NonConvex", 2)
@variable(model, lower_bounds[i] <= a[i=1:nv] <= upper_bounds[i])
@objective(model, Max, dot(hat_theta,a))
#@variable(model, u)
#@objective(model, Max, dot(hat_theta,a) + sqrt(beta)*u)
#@constraint(model, transpose(a)*inv_V*a == u^2)
optimize!(model)
println(termination_status(model))

alpha_vector = value.(a)
alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
reward = run_then_return_val_loss(deepcopy(data_area),alpha_config,initial_config,optimizer,initial_iters)

println("Iterations to converge: ", 200 - reward)