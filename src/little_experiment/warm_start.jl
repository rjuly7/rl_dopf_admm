#path = "/storage/scratch1/8/rharris94/rl_dopf_admm"
path = "C:/Users/User/Documents/rl_dopf_admm/"

using Pkg
Pkg.activate(".")
using Distributed 
using Flux
using Flux: @epochs
using Flux.Optimise: update!
using Flux.Losses: mse
using Plots
using Statistics
using Random 
using StatsBase 
using Ipopt 
using PowerModels 
using PowerModelsADA 
using BSON 
using BSON: @save 
include("$path/src/little_experiment/data_collection.jl")

println(pwd() )
include("$path/src/little_experiment/nn_functions.jl")
casename = "case118_3"
nn_num = 1
number_epochs = 3
batchsize = 4000
need_csv = 0
n_areas = 3
println(ARGS)

if lastindex(ARGS) >= 4
    casename = ARGS[1]
    nn_num = parse(Int32,ARGS[2])
    number_epochs = parse(Int32,ARGS[3])
    batchsize = parse(Int32,ARGS[4])
else
    println("Using default, no command line arguments given")
end

case_path = "data/$casename.m"
data = parse_file(case_path)
if (need_csv == 1)
    partition_path= "data/$casename"*"_$n_areas.csv"
    assign_area!(data, partition_path)
end
initial_iters = 20

trainsize = 0.9 

maps = BSON.load("data/little_experiment/$casename"*"_maps.bson")
primal_map = maps["primal"]
dual_map = maps["dual"]
load_map = maps["load"]

nn_primal = BSON.load("$path/data/little_experiment/$casename"*"_nn$nn_num"*"_primal.bson")[:nn]
nn_dual = BSON.load("$path/data/little_experiment/$casename"*"_nn$nn_num"*"_dual.bson")[:nn]
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods
tol = 1e-4 
du_tol = 0.1 
max_iteration = 600
data_area,load_vector = get_perturbed_data_area(deepcopy(data),model_type,dopf_method,tol,du_tol,max_iteration,load_map)

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

n_repeat = 3
#n_iters_repeat = @distributed (append!) for repeat=1:n_repeat 
n_iters_repeat = []
n_iters_repeat_p = Dict("1" => [], "2" => [], "3" => [])
for i=1:n_repeat 
    data_area,load_vector = get_perturbed_data_area(deepcopy(data),model_type,dopf_method,tol,du_tol,max_iteration,load_map)
    primal_out = nn_primal(load_vector)
    dual_out = nn_dual(load_vector)
    data_area = warm_start(data_area,primal_map,dual_map,vcat(primal_out,dual_out))

    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    vars_vector,iteration = run_iterations(initial_config,initial_config,deepcopy(data_area),ACPPowerModel,adaptive_admm_methods,primal_map,dual_map,initial_iters)

    println("Iterations to converge: ", iteration)
    push!(n_iters_repeat, iteration)
    primal_end = Int(length(vars_vector)/2)
    println("primal end: ", primal_end)
    primal_true = vars_vector[1:primal_end]
    dual_true = vars_vector[primal_end+1:end]
    println("primal loss ", Flux.Losses.mse(primal_out,primal_true), " dual loss: ", Flux.Losses.mse(dual_out,dual_true))

    for j=1:3
        new_start = []
        for vidx in eachindex(primal_out) 
            if primal_out[vidx] > primal_true[vidx]
                push!(new_start, primal_out[vidx] + rand()*j*1e-2)
            elseif primal_out[vidx] <= primal_true[vidx] 
                push!(new_start, primal_out[vidx] - rand()*j*1e-2)
            end
        end
        for vidx in eachindex(dual_out) 
            if dual_out[vidx] > dual_true[vidx]
                push!(new_start, dual_out[vidx] + rand()*j*8)
            elseif dual_out[vidx] <= dual_true[vidx]
                push!(new_start, dual_out[vidx] - rand()*j*8)
            end
        end
        println("new loss ", Flux.Losses.mse(new_start[1:primal_end],primal_true), " new dual loss: ", Flux.Losses.mse(new_start[primal_end+1:end],dual_true))
        data_area = warm_start(data_area,primal_map,dual_map,new_start)

        optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
        vars_vector,iteration = run_iterations(initial_config,initial_config,deepcopy(data_area),ACPPowerModel,adaptive_admm_methods,primal_map,dual_map,initial_iters)
        push!(n_iters_repeat_p[string(j)],iteration)
    end
end

bson("$path/data/little_experiment/$casename"*"_convresults.bson", Dict("n_iters_repeat" => n_iters_repeat, "n_iters_repeat_p" => n_iters_repeat_p))

stuff = BSON.load("$path/data/little_experiment/case118_3_convresults.jl")
n = stuff["n_iters_repeat"]
np = stuff["n_iters_repeat_p"]

plot([n,np["1"],np["2"],np["3"]],label=["nn" "p1" "p2" "p3"])

plot([n,np["3"]],label=["nn" "p3"])

