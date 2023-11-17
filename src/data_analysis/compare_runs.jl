include("C:/Users/User/Documents/rl_dopf_admm/src/adaptive/adaptive_base_functions.jl")
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods
max_iteration = 500
tol = 1e-4 
du_tol = 0.1 
baseline_alpha_pq = 400
baseline_alpha_vt =  4000
tau_inc = 0.007
tau_dec = 0.007 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol, baseline_alpha_pq, baseline_alpha_vt)
data_area = run_to_end(data_area::Dict{Int,Any}, tau_inc, tau_dec)