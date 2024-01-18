sqnorm(x) = sum(abs2, x);

function loss(X, Y, λ, m, params)
    return Flux.Losses.mse(m(X),Y) + λ*sum(sqnorm, params)
end

function my_custom_train!(params, X, Y, opt, loss_tr, batchsize, λ, model)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    idcs = sample(axes(Y,2), batchsize, replace=false)
    Xm = X[:,idcs]
    Ym = Y[:,idcs]
    local training_loss
    gs = gradient(params) do
        training_loss = Flux.Losses.mse(model(Xm),Ym) + λ*sum(sqnorm, params)
        return training_loss
    end
    update!(opt, params, gs)
    push!(loss_tr, training_loss);
end

function my_custom_train_logloss!(params, X, Y, opt, loss_tr, batchsize, λ, model)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    idcs = sample(1:length(axes(Y,2)), batchsize, replace=false)
    Xm = X[:,idcs]
    Ym = Y[:,idcs]
    local training_loss
    gs = gradient(params) do
        training_loss = Flux.Losses.msle(model(Xm),Ym) + λ*sum(sqnorm, params)
        return training_loss
    end
    update!(opt, params, gs)
    push!(loss_tr, training_loss);
end

function get_dataset(data_path, trainsize)
    f = data_path[1]
    input = BSON.load(f)["X"]
    output = BSON.load(f)["y"]
    for f in data_path[2:end]
        X = BSON.load(f)["X"]
        input = hcat(input,X)
        Y = BSON.load(f)["y"]
        output = hcat(output,Y)
    end
    n_samples = length(axes(input,2))
    train_len = Int(ceil(trainsize*n_samples))
    train_idcs = sample(1:n_samples, train_len, replace=false)
    test_idcs = setdiff(1:n_samples, train_idcs)  
    Xtrain = input[:,train_idcs]
    Xtest = input[:,test_idcs]
    ytrain = output[:,train_idcs]
    ytest = output[:,test_idcs]  
    return Xtrain, ytrain, Xtest, ytest 
end

function warm_start(data_area,primal_map,dual_map,nn_pred)
    for area in keys(data_area)
        for n in keys(data_area[area]["shared_variable"])
            for v in keys(data_area[area]["shared_variable"][n])
                for k in keys(data_area[area]["shared_variable"][n][v])
                    data_area[area]["shared_variable"][n][v][k] = nn_pred[primal_map[area][n][v][k]]
                    data_area[area]["received_variable"][n][v][k] = nn_pred[primal_map[area][n][v][k]]
                    if area < parse(Int,n)
                        data_area[area]["dual_variable"][n][v][k] = nn_pred[dual_map[area][n][v][k]] 
                    else
                        data_area[area]["dual_variable"][n][v][k] = -nn_pred[dual_map[parse(Int,n)][string(area)][v][k]]
                    end
                end
            end
        end
    end
    return data_area 
end


function train_primal(n_epochs, casename, batchsize, adp)
    trainsize = 0.9 

    file_list = readdir("$path/data/new_approach")
    relevant_idcs = findall(x -> contains(x, "$casename"*"_dataset"), file_list)
    dataset_list = file_list[relevant_idcs]
    filenames = Vector{String}(undef, length(dataset_list))
    for n in eachindex(dataset_list)
        filenames[n] = "$path/data/new_approach/"*dataset_list[n]
    end   

    Xtrain, ytrain, Xtest, ytest = get_dataset(filenames,trainsize)
    n_out = length(axes(ytrain,1))
    primal_end = Int(n_out/2)
    ytrain = ytrain[1:primal_end,:]
    ytest = ytest[1:primal_end,:]

    println(size(Xtrain), "  ", size(ytrain))

    n_in = length(axes(Xtrain,1))
    n_out = length(axes(ytrain,1))
    n_mid = n_in*2 

    model = Chain(Dense(n_in,n_mid,relu), Dense(n_mid,n_mid,relu), Dense(n_mid,n_mid,relu), Dense(n_mid,n_mid,relu), Dense(n_mid,n_out,identity))

    opt = ADAM(adp)  
    loss_tr = Vector{Float32}()
    val_tr = Vector{Float32}()
    params = Flux.params(model)
    lam = 0

    for n = 1:n_epochs
        my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, lam, model)
        push!(val_tr, Flux.Losses.mse(model(Xtest),ytest))
        if mod(n,100) == 0
            println("Epoch ", n, ", Loss ", loss_tr[end])
        end
    end

    test_loss = Flux.Losses.mse(model(Xtest),ytest)
    println("Test loss: ", test_loss)

    return model, loss_tr, val_tr, test_loss 
end

function train_primal(n_epochs, casename, batchsize, model, adp)
    trainsize = 0.9 

    file_list = readdir("$path/data/new_approach")
    relevant_idcs = findall(x -> contains(x, "$casename"*"_dataset"), file_list)
    dataset_list = file_list[relevant_idcs]
    filenames = Vector{String}(undef, length(dataset_list))
    for n in eachindex(dataset_list)
        filenames[n] = "$path/data/new_approach/"*dataset_list[n]
    end   

    Xtrain, ytrain, Xtest, ytest = get_dataset(filenames,trainsize)
    n_out = length(axes(ytrain,1))
    primal_end = Int(n_out/2)
    ytrain = ytrain[1:primal_end,:]
    ytest = ytest[1:primal_end,:]

    println(size(Xtrain), "  ", size(ytrain))

    n_in = length(axes(Xtrain,1))
    n_out = length(axes(ytrain,1))
    n_mid = n_in*2 

    opt = ADAM(adp)  
    loss_tr = Vector{Float32}()
    val_tr = Vector{Float32}()
    params = Flux.params(model)
    lam = 0

    for n = 1:n_epochs
        my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, lam, model)
        push!(val_tr, Flux.Losses.mse(model(Xtest),ytest))
        if mod(n,100) == 0
            println("Epoch ", n, ", Loss ", loss_tr[end])
        end
    end

    test_loss = Flux.Losses.mse(model(Xtest),ytest)
    println("Test loss: ", test_loss)

    return model, loss_tr, val_tr, test_loss 
end

function train_dual(n_epochs, casename, batchsize, adp)
    trainsize = 0.9 

    file_list = readdir("$path/data/new_approach")
    relevant_idcs = findall(x -> contains(x, "$casename"*"_dataset"), file_list)
    dataset_list = file_list[relevant_idcs]
    filenames = Vector{String}(undef, length(dataset_list))
    for n in eachindex(dataset_list)
        filenames[n] = "$path/data/new_approach/"*dataset_list[n]
    end   

    Xtrain, ytrain, Xtest, ytest = get_dataset(filenames,trainsize)
    n_out = length(axes(ytrain,1))
    primal_end = Int(n_out/2)
    ytrain = ytrain[primal_end+1:end,:]
    ytest = ytest[primal_end+1:end,:]

    println(size(Xtrain), "  ", size(ytrain))

    n_in = length(axes(Xtrain,1))
    n_out = length(axes(ytrain,1))
    n_mid = n_in*2 

    model = Chain(Dense(n_in,n_mid,relu), Dense(n_mid,n_mid,relu), Dense(n_mid,n_mid,relu), Dense(n_mid,n_mid,relu), Dense(n_mid,n_out,identity))

    opt = ADAM(adp)  
    loss_tr = Vector{Float32}()
    val_tr = Vector{Float32}()
    params = Flux.params(model)
    lam = 0

    for n = 1:n_epochs
        my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, lam, model)
        push!(val_tr, Flux.Losses.mse(model(Xtest),ytest))
        if mod(n,100) == 0
            println("Epoch ", n, ", Loss ", loss_tr[end])
        end
    end

    test_loss = Flux.Losses.mse(model(Xtest),ytest)
    println("Test loss: ", test_loss)

    return model, loss_tr, val_tr, test_loss 
end

function train_dual(n_epochs, casename, batchsize, model, adp)
    trainsize = 0.9 

    file_list = readdir("$path/data/new_approach")
    relevant_idcs = findall(x -> contains(x, "$casename"*"_dataset"), file_list)
    dataset_list = file_list[relevant_idcs]
    filenames = Vector{String}(undef, length(dataset_list))
    for n in eachindex(dataset_list)
        filenames[n] = "$path/data/new_approach/"*dataset_list[n]
    end   

    Xtrain, ytrain, Xtest, ytest = get_dataset(filenames,trainsize)
    n_out = length(axes(ytrain,1))
    primal_end = Int(n_out/2)
    ytrain = ytrain[primal_end+1:end,:]
    ytest = ytest[primal_end+1:end,:]

    println(size(Xtrain), "  ", size(ytrain))

    n_in = length(axes(Xtrain,1))
    n_out = length(axes(ytrain,1))
    n_mid = n_in*2 

    opt = ADAM(adp)  
    loss_tr = Vector{Float32}()
    val_tr = Vector{Float32}()
    params = Flux.params(model)
    lam = 0

    for n = 1:n_epochs
        my_custom_train!(params, Xtrain, ytrain, opt, loss_tr, batchsize, lam, model)
        push!(val_tr, Flux.Losses.mse(model(Xtest),ytest))
        if mod(n,100) == 0
            println("Epoch ", n, ", Loss ", loss_tr[end])
        end
    end

    test_loss = Flux.Losses.mse(model(Xtest),ytest)
    println("Test loss: ", test_loss)

    return model, loss_tr, val_tr, test_loss 
end