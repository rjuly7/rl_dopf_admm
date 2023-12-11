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