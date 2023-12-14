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
using BSON 
using BSON: @save 
include("$path/src/little_experiment/data_collection.jl")

println(pwd() )
include("$path/src/little_experiment/nn_functions.jl")
casename = "case118_3"
nn_num = 1
number_epochs = 3
batchsize = 4000

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

file_list = readdir("$path/data/little_experiment")
relevant_idcs = findall(x -> contains(x, "$casename"*"_dataset"), file_list)
dataset_list = file_list[relevant_idcs]
filenames = Vector{String}(undef, length(dataset_list))
for n in eachindex(dataset_list)
    filenames[n] = "$path/data/little_experiment/"*dataset_list[n]
end   

Xtrain, ytrain, Xtest, ytest = get_dataset(filenames,trainsize)

println(size(Xtrain))

n_in = length(axes(Xtrain,1))
n_out = length(axes(ytrain,1))
n_mid = n_in*2 

nn_primal = BSON.load("$path/data/little_experiment/$casename"*"_nn$nn_num"*"_primal.bson")[:nn]
nn_dual = BSON.load("$path/data/little_experiment/$casename"*"_nn$nn_num"*"_dual.bson")[:nn]
data_area = get_perturbed_data_area(deepcopy(data))

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)
maps = BSON.load("data/little_experiment/$casename"*"_maps.bson")
primal_map = maps["primal"]
dual_map = maps["dual"]
load_map = maps["load"]

n_repeat = 100
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

bson("$path/data/little_experiment/$casename"*"_convresults.jl", Dict("n_iters_repeat" => n_iters_repeat, "n_iters_repeat_p" => n_iters_repeat_p))