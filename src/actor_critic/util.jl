function sigmoid_norm_primal(X)
    return @. 1 / (1 + exp(-2*X))
end

function sigmoid_norm_dual(X)
    return @. 1 / (1 + exp(-0.3*X))
end

function tanh_norm(x)
    return tanh.(x)
end