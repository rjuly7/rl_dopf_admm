function get_action_idcs(a,n_actions)
    per_eta_inc = n_actions[2]*n_actions[3]*n_actions[4]
    eta_inc_idx = Int(ceil(a/per_eta_inc))
    per_eta_dec = n_actions[3]*n_actions[4]
    eta_dec_idx = Int(ceil((a - per_eta_inc*(eta_inc_idx-1))/per_eta_dec))
    per_mu_inc = n_actions[4]
    mu_inc_idx = Int(ceil((a - per_eta_inc*(eta_inc_idx-1) - per_eta_dec*(eta_dec_idx-1))/per_mu_inc))
    mu_dec_idx = a - per_eta_inc*(eta_inc_idx-1) - per_eta_dec*(eta_dec_idx-1) - per_mu_inc*(mu_inc_idx - 1)
    return [eta_inc_idx,eta_dec_idx,mu_inc_idx,mu_dec_idx]
end

function sigmoid_norm_primal(X)
    return @. 1 / (1 + exp(-2*X))
end

function sigmoid_norm_dual(X)
    return @. 1 / (1 + exp(-0.3*X))
end