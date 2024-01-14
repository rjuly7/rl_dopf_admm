#path = "/storage/scratch1/8/rharris94/rl_dopf_admm"
path = "C:/Users/User/Documents/rl_dopf_admm/"

using Pkg
Pkg.activate(".")
using Distributed 
using PowerModelsADA 
using Ipopt 
using PowerModels 
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

println(pwd() )
include("$path/src/new_approach/nn_functions.jl")
include("$path/src/new_approach/data_collection.jl")

casename = "case118_3"
need_csv = 0 
println(ARGS)

if lastindex(ARGS) >= 2
    casename = ARGS[1]
    need_csv = ARGS[2]
else
    println("Using default, no command line arguments given")
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

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

primal_map, dual_map,load_map = create_maps(data,data_area)
alpha_pq = 400
alpha_vt = 4000
alpha_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

#Start by generating dataset of 1000
run_num = 1
n_iters = 60
n_runs = 30
X,y = gen_dataset(n_runs,n_iters,data,alpha_config) 
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
nn_num = 1
n_epochs = 20 
batchsize = Int(n_runs/2) 
adp = 1e-3 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
#Train dual NN for some epochs 
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))

#Next, generate dataset of 1000 
run_num = 2
n_iters = 100
n_runs = 30
X,y = gen_dataset(n_runs,n_iters,data,alpha_config,primal_map,dual_map,model_primal,model_dual)
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
nn_num = 2
n_epochs = 50 
batchsize = Int(n_runs/2) 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_primal, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
model_dual, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_dual, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
