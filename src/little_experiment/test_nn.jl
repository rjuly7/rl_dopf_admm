#path = "/storage/scratch1/8/rharris94/rl_dopf_admm"
path = "C:/Users/User/Documents/rl_dopf_admm/"

using Pkg
Pkg.activate(".")
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
include("$path/src/little_experiment/nn_functions.jl")

nn_num = 1
casename = "case118_3"
NN_stuff = BSON.load("$path/data/little_experiment/nn_$nn_num.bson")
NN = NN_stuff[:nn]

trainsize = 0.9 

file_list = readdir("$path/data/little_experiment")
relevant_idcs = findall(x -> contains(x, "$casename"*"_dataset"), file_list)
dataset_list = file_list[relevant_idcs]
filenames = Vector{String}(undef, length(dataset_list))
for n in eachindex(dataset_list)
    filenames[n] = "$path/data/little_experiment/"*dataset_list[n]
end   

Xtrain, ytrain, Xtest, ytest = get_dataset(filenames,trainsize)

pred = NN(Xtest)
test_loss = Flux.Losses.mse(pred,ytest)
