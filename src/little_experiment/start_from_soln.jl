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
using LinearAlgebra
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

# file_list = readdir("$path/data/little_experiment")
# relevant_idcs = findall(x -> contains(x, "$casename"*"_dataset"), file_list)
# dataset_list = file_list[relevant_idcs]
# filenames = Vector{String}(undef, length(dataset_list))
# for n in eachindex(dataset_list)
#     filenames[n] = "$path/data/little_experiment/"*dataset_list[n]
# end   

# Xtrain, ytrain, Xtest, ytest = get_dataset(filenames,trainsize)
maps = BSON.load("data/little_experiment/$casename"*"_maps.bson")
primal_map = maps["primal"]
dual_map = maps["dual"]
load_map = maps["load"]

nn_primal = BSON.load("$path/data/little_experiment/$casename"*"_nn$nn_num"*"_primal.bson")[:nn]
nn_dual = BSON.load("$path/data/little_experiment/$casename"*"_nn$nn_num"*"_dual.bson")[:nn]
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods
tol = 1e-4
du_tol = 1e-4
max_iteration = 2000
data_area,load_vector = get_perturbed_data_area(deepcopy(data),model_type,dopf_method,tol,du_tol,max_iteration,load_map)

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

n_repeat = 6
p_list = 1:4
#n_iters_repeat = @distributed (append!) for repeat=1:n_repeat 
n_iters_repeat = []
n_iters_repeat_p = Dict(string(p) => [] for p in p_list)
new_start_p = Dict(string(p) => [] for p in p_list)
data_area_p = []
for i=1:n_repeat 
    data_area,load_vector = get_perturbed_data_area(deepcopy(data),model_type,dopf_method,tol,du_tol,max_iteration,load_map)
    push!(data_area_p,deepcopy(data_area))
    primal_out = nn_primal(load_vector)
    dual_out = nn_dual(load_vector)
    #data_area = warm_start(data_area,primal_map,dual_map,vcat(primal_out,dual_out))

    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    vars_vector,iteration = run_iterations(initial_config,initial_config,deepcopy(data_area),ACPPowerModel,adaptive_admm_methods,primal_map,dual_map,initial_iters)

    println("Iterations to converge: ", iteration)
    push!(n_iters_repeat, iteration)
    primal_end = Int(length(vars_vector)/2)
    println("primal end: ", primal_end)
    primal_true = vars_vector[1:primal_end]
    dual_true = vars_vector[primal_end+1:end]

    for j in p_list 
        new_start = []
        for vidx in eachindex(primal_true) 
            push!(new_start, primal_true[vidx] + (rand()-0.5)*j*1e-2)
        end
        for vidx in eachindex(dual_true) 
            push!(new_start, dual_true[vidx] + (rand()-0.5)*j*8)
        end
        println("new loss ", Flux.Losses.mse(new_start[1:primal_end],primal_true), " new dual loss: ", Flux.Losses.mse(new_start[primal_end+1:end],dual_true))
        data_area = warm_start(data_area,primal_map,dual_map,new_start)

        optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
        vars_vector,iteration = run_iterations(initial_config,initial_config,deepcopy(data_area),ACPPowerModel,adaptive_admm_methods,primal_map,dual_map,initial_iters)
        push!(n_iters_repeat_p[string(j)],iteration)
        push!(new_start_p[string(j)],deepcopy(new_start))
    end
end

bson("$path/data/little_experiment/$casename"*"_start_from_soln_results.bson", Dict("n_iters_repeat" => n_iters_repeat, "n_iters_repeat_p" => n_iters_repeat_p, "new_start_p" => new_start_p, "data_area_p" => data_area_p))


ii = 1
data_area = deepcopy(data_area_p[ii])

optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
vars_vector0,iteration0,resid0 = run_iterations_track_resid(initial_config,initial_config,data_area,ACPPowerModel,adaptive_admm_methods,primal_map,dual_map,initial_iters)
println(data_area[1]["received_variable"]["2"]["va"]["24"], "  ", data_area[1]["shared_variable"]["2"]["va"]["24"])
println(data_area[1]["received_variable"]["2"]["qf"]["111"], "  ", data_area[1]["shared_variable"]["2"]["qf"]["111"])
cost0 = calc_dist_gen_cost(data_area)

new_start0 = new_start_p["1"][ii]

println("diff: ", LinearAlgebra.norm((new_start0-vars_vector0),2))

warm_data_area = warm_start(deepcopy(data_area_p[ii]),primal_map,dual_map,new_start0)

optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
vars_vector1,iteration1,resid1 = run_iterations_track_resid(initial_config,initial_config,warm_data_area,ACPPowerModel,adaptive_admm_methods,primal_map,dual_map,initial_iters)
println(warm_data_area[1]["received_variable"]["2"]["va"]["24"], "  ", warm_data_area[1]["shared_variable"]["2"]["va"]["24"])
println(warm_data_area[1]["received_variable"]["2"]["qf"]["111"], "  ", warm_data_area[1]["shared_variable"]["2"]["qf"]["111"])
cost1 = calc_dist_gen_cost(warm_data_area)

new_start1 = new_start_p["4"][ii]

println("diff: ", LinearAlgebra.norm((new_start1-vars_vector1),2))

warm_data_area = warm_start(data_area,primal_map,dual_map,new_start1)

optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
vars_vector2,iteration2,resid2 = run_iterations_track_resid(initial_config,initial_config,deepcopy(warm_data_area),ACPPowerModel,adaptive_admm_methods,primal_map,dual_map,initial_iters)


LinearAlgebra.norm([value for area in keys(mismatch) if area != area_id for variable in keys(mismatch[area]) for (idx,value) in mismatch[area][variable]], 2)
