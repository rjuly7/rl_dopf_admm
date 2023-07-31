stuff = env.rewardPerStep 
rewards = []
for i in eachindex(stuff)
    push!(rewards, stuff[i]["reward"])
end
include("base_functions.jl")
data_area = stuff[47]["data_area"]
iteration = stuff[47]["iteration"]
alpha_pq = stuff[47]["pq"]
alpha_vt = stuff[47]["vt"]
base_residual_data, agent_base_residual_data, base_data_area, base_iteration, base_converged = run_some_iterations(deepcopy(data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(iteration), env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, rng);
pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged = run_some_iterations(deepcopy(data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(iteration), alpha_pq, alpha_vt, env.params.n_history, rng);

plot([pol_residual_data["primal"],pol_residual_data["dual"]], label=["primal" "dual"], yscale=:log10)

push!(env.rewardPerStep,Dict("reward" => env.reward, "pq" => alpha_pq, "vt" => alpha_vt, "state" => env.state, "data_area" => save_data_area, "iteration" => save_iteration))

data_area, alpha_trace = quick_adaptive_test(data)

pq = []
vt = [] 
for iter =1:length(alpha_trace[1])
    vals = alpha_trace[1][iter]["2"]
    for var_type in keys(vals)
        for (b,alph) in vals[var_type]
            if var_type == "pt" || var_type == "pf" || var_type == "qf" || var_type == "qt"
                push!(pq, alph)
            else
                push!(vt,alph)
            end
        end
    end
end