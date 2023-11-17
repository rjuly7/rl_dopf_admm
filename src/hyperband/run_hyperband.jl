using Pkg 
Pkg.activate(".")

using PowerModelsADA
using Ipopt 
using StatsBase 
include("hyperband_functions.jl")

case_path = "data/case118_3.m"
data = parse_file(case_path)
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 10000
optimizer = Ipopt.Optimizer 

R = 5
eta = 3 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

pq_bounds = [150,800]
vt_bounds = [3000,5000]

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

function run_hyperband_perturb_loads(R,eta,data_area,pq_bounds,vt_bounds,initial_config,optimizer)
    smax = Int(floor(log(eta,R)))
    B = (smax + 1)*R 
    alpha_configs = []
    top_idcs_vals = []
    for s = smax:-1:0
        n = Int(ceil(B/R*eta^s/(s+1)))
        r = R/(eta^s)
        println("s: ", s, "  n: ", n)
        alpha_configs = get_hyperparameter_configuration(n,data_area,pq_bounds,vt_bounds)
        for i = 0:s
            n_i = Int(floor(n/(eta^i)))
            r_i = r*eta^i 
            println("s: ", s, "  n: ", n, " i: ", i, " n_i: ", n_i, " r_i: ", r_i)
            loss = []
            count = 1 
            for alpha in alpha_configs 
                track_alpha_loss = []
                for rr=1:r_i 
                    println("Count: ", count, "  rr: ", rr)
                    push!(track_alpha_loss, run_then_return_val_loss(deepcopy(data_area),alpha,initial_config,optimizer))
                end
                push!(loss, mean(track_alpha_loss))
                count += 1
            end
            println("K: ", floor(n_i/eta))
            alpha_configs, top_idcs_vals = top_k(alpha_configs,loss,floor(n_i/eta))
        end
    end 
    return alpha_configs, top_idcs_vals 
end

function run_hyperband(R,eta,data_area,pq_bounds,vt_bounds,initial_config,optimizer)
    total_runs = get_run_count(eta,R)
    alpha_configs = get_hyperparameter_configuration(n,data_area,pq_bounds,vt_bounds)
    for i = 0:s
        n_i = Int(floor(n/(eta^i)))
        r_i = r*eta^i 
        println("s: ", s, "  n: ", n, " i: ", i, " n_i: ", n_i, " r_i: ", r_i)
        loss = []
        count = 1 
        for alpha in alpha_configs 
            track_alpha_loss = []
            for rr=1:r_i 
                println("Count: ", count, "  rr: ", rr)
                push!(track_alpha_loss, run_then_return_val_loss(deepcopy(data_area),alpha,initial_config,optimizer))
            end
            push!(loss, mean(track_alpha_loss))
            count += 1
        end
        println("K: ", floor(n_i/eta))
        alpha_configs, top_idcs_vals = top_k(alpha_configs,loss,Int(floor(n_i/eta)))
    end
    println("Length: ", length(alpha_configs[i]))
    return alpha_configs, top_idcs_vals 
end

final_configs, top_idcs = run_hyperband(R,eta,data_area,pq_bounds,vt_bounds,initial_config,optimizer)