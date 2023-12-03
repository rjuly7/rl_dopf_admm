using Distributed 
using Gurobi 
using LinearAlgebra
using Ipopt 
using JuMP 
using BSON 
using Plots 
using PowerModelsADA 
include("linucb_functions.jl")

#case118 
pqs_l = [100, 100, 100, 150, 200, 250, 300, 350, 375]
pqs_u = [3200, 2800, 2000, 1200, 900, 700, 600, 500, 450]

vts_l = [1400, 1600, 2000, 2400, 2800, 3200, 3600, 3800, 3900]
vts_u = [7000, 6500, 6000, 5600, 5200, 4800, 4400, 4200, 4100]

#case30
pqs_l = [100, 100, 150, 200, 250, 300, 350]
pqs_u = [2800, 2000, 1200, 900, 700, 600, 500]

vts_l = [1600, 2000, 2400, 2800, 3200, 3600, 3800]
vts_u = [6500, 6000, 5600, 5200, 4800, 4400, 4200]

run_num = 1
casename = "case30"
need_csv = 1

n_iters = []
for b_num in eachindex(pqs_l)
    n_areas = 3
    pq_lower = pqs_l[b_num]
    pq_upper = pqs_u[b_num]
    vt_lower = vts_l[b_num]
    vt_upper = vts_u[b_num]
    stuff = BSON.load("data/hyperband/linucb_$casename"*"_$pq_lower"*"_$pq_upper"*"_$vt_lower"*"_$vt_upper"*"_$run_num.jl")
    trace_params = stuff["trace"]

    hat_thetas = []
    for i in eachindex(trace_params["V"])
        V = trace_params["V"][i]
        y = trace_params["y"][i]
        push!(hat_thetas, inv(V)*y)
    end

    # plot(trace_params["reward"][200:end])

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

    pq_bounds = [pq_lower,pq_upper]
    vt_bounds = [vt_lower,vt_upper]

    alpha_pq = 400
    alpha_vt = 4000 
    initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

    lower_bounds,upper_bounds = get_bounds(pq_bounds,vt_bounds,data_area)
    alpha_config,alpha_vector = get_hyperparameter_configuration(deepcopy(data_area),pq_bounds,vt_bounds) #pull initial config 
    nv = length(alpha_vector)
    T = 500 
    beta = 1 + sqrt(2*log(T)+nv*log((nv+T)/nv))

    # model = Model(Ipopt.Optimizer)
    # #set_optimizer_attribute(model, "NonConvex", 2)
    # @variable(model, lower_bounds[i] <= a[i=1:nv] <= upper_bounds[i])
    # #@objective(model, Max, dot(hat_theta,a))
    # @variable(model, u)
    # @objective(model, Max, dot(hat_theta,a) + sqrt(beta)*u)
    # @constraint(model, transpose(a)*inv_V*a == u^2)
    # optimize!(model)
    # println(termination_status(model))

    # alpha_vector = value.(a)
    # alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
    # initial_iters = 20
    # reward = run_then_return_val_loss(deepcopy(data_area),alpha_config,initial_config,optimizer,initial_iters)

    # println("Iterations to converge: ", 200 - reward)

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

    push!(n_iters, 200-reward)
    println("Iterations to converge: ", 200 - reward)
end

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

baseline_r = run_then_return_val_loss(deepcopy(data_area),initial_config,initial_config,optimizer,initial_iters)
println("Iterations to converge: ", 200 - reward)
baseline_iter = 200 - baseline_r 
spread = pqs_u - pqs_l + vts_u - vts_l 
plot(spread,n_iters,ylim=(0,baseline_iter + 10),linewidth=2,label="LinUCB",xlabel="Length of search space",ylabel="Iterations to converge")
plot!(spread,baseline_iter*ones(length(spread)),linewidth=2,label="Baseline")
savefig("data/figs/$casename"*"_iters_vs_space.png")

plot(spread, pqs_l, fillrange = pqs_u, fillalpha = 0.35, label = "PQ values",xlabel="Length of search space", ylabel="Search values")
plot!(spread, vts_l, fillrange = vts_u, fillalpha = 0.35, label = "VT values")
plot!(spread,4000*ones(length(spread)),line=:dash,linewidth=2,color="black",label="Baseline PQ")
plot!(spread,400*ones(length(spread)),line=:dot,linewidth=2,color="black",label="Baseline VT")
savefig("data/figs/$casename"*"_bounds.png")