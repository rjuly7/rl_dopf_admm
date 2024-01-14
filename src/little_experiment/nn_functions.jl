sqnorm(x) = sum(abs2, x);

function loss(X, Y, λ, m, params)
    return Flux.Losses.mse(m(X),Y) + λ*sum(sqnorm, params)
end

function my_custom_train!(params, X, Y, opt, loss_tr, batchsize, λ, model)
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    idcs = sample(1:length(axes(Y,2)), batchsize, replace=false)
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

function bad_warm_start(data_area,primal_map,dual_map,nn_pred)
    for area in keys(data_area)
        for n in keys(data_area[area]["shared_variable"])
            for v in keys(data_area[area]["shared_variable"][n])
                for k in keys(data_area[area]["shared_variable"][n][v])
                    data_area[area]["shared_variable"][n][v][k] = nn_pred[primal_map[area][n][v][k]]
                    data_area[area]["received_variable"][n][v][k] = nn_pred[primal_map[area][n][v][k]]
                    data_area[area]["dual_variable"][n][v][k] = nn_pred[dual_map[area][n][v][k]] 
                end
            end
        end
    end
    return data_area 
end