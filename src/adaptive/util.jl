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