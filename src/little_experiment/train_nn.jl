path = "/storage/scratch1/8/rharris94/rl_dopf_admm"
#path = "C:/Users/User/Documents/rl_dopf_admm/"

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
n_mid = n_in + 20

model = Chain(Dense(n_in,n_in,relu), Dense(n_in,n_in,relu), Dense(n_in,n_in,relu), Dense(n_in,n_out,identity))

opt = ADAM(5e-4)  
loss_tr = Vector{Float32}()
val_tr = Vector{Float32}()
params = Flux.params(model)
lam = 0

for n = 1:number_epochs
    my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, lam, model)
    push!(val_tr, Flux.Losses.mse(model(Xtest),ytest))
    if mod(n,100) == 0
        println("Epoch ", n, ", Loss ", loss_tr[end])
    end
end

test_loss = Flux.Losses.mse(model(Xtest),ytest)
println("Test loss: ", test_loss)

pred = model(Xtest)

errors = []
for i in axes(pred,1)
    for j in axes(pred,2)
#        push!(errors, (pred[i,j]-ytest[i,j])^2)
    push!(errors, (pred[i,j]-ytest[i,j]))
    end
end

bson("$path/data/little_experiment/nn_$nn_num.bson", Dict(:nn => model, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr, :errors => errors))
