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

casename = "pglib_opf_case588_sdet_8"
n_epochs = 20 
batchsize = Int(n_runs/2) 
adp = 1e-3 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
#Train dual NN for some epochs 
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
