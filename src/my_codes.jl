# stuff = env.rewardPerStep 
# rewards = []
# for i in eachindex(stuff)
#     push!(rewards, stuff[i]["reward"])
# end
# include("base_functions.jl")
# data_area = stuff[47]["data_area"]
# iteration = stuff[47]["iteration"]
# alpha_pq = stuff[47]["pq"]
# alpha_vt = stuff[47]["vt"]
# base_residual_data, agent_base_residual_data, base_data_area, base_iteration, base_converged = run_some_iterations(deepcopy(data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(iteration), env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, rng);
# pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged = run_some_iterations(deepcopy(data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(iteration), alpha_pq, alpha_vt, env.params.n_history, rng);

# plot([pol_residual_data["primal"],pol_residual_data["dual"]], label=["primal" "dual"], yscale=:log10)

# push!(env.rewardPerStep,Dict("reward" => env.reward, "pq" => alpha_pq, "vt" => alpha_vt, "state" => env.state, "data_area" => save_data_area, "iteration" => save_iteration))

# data_area, alpha_trace = quick_adaptive_test(data)

# pq = []
# vt = [] 
# for iter =1:length(alpha_trace[1])
#     vals = alpha_trace[1][iter]["2"]
#     for var_type in keys(vals)
#         for (b,alph) in vals[var_type]
#             if var_type == "pt" || var_type == "pf" || var_type == "qf" || var_type == "qt"
#                 push!(pq, alph)
#             else
#                 push!(vt,alph)
#             end
#         end
#     end
# end

using Pkg 
Pkg.activate(".")
using PowerModelsADA
using Ipopt 
using Suppressor

function quick_adaptive_test(data, dual_measure=false;
    dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000, tol=1e-4) 
    
    areas_id = get_areas_id(data)
    # get areas ids
    ## decompose the system into subsystems
    data_area = decompose_system(data)

    # initilize distributed power model parameters
    for area in areas_id
        #dopf_method.initialize_method(data_area[area], model_type; max_iteration=max_iteration, tol=tol)
        dopf_method.initialize_method(data_area[area], model_type; termination_measure= "mismatch_dual_residual", max_iteration=max_iteration, tol=tol)
    end

    # initialize the algorithms global counters
    iteration = 1
    flag_convergence = false
    alpha_trace = Dict(i => [] for i in areas_id)
    # start iteration
    while iteration < max_iteration && !flag_convergence

        for i in areas_id 
            push!(alpha_trace[i], data_area[i]["alpha"])
        end
        # solve local problem and update solution
        info = @capture_out begin
            for area in areas_id
                result = solve_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
                update_data!(data_area[area], result["solution"])
            end
        end


        # share solution with neighbors, the shared data is first obtained to facilitate distributed implementation
        for area in areas_id # sender subsystem
            for neighbor in data_area[area]["neighbors"] # receiver subsystem
                shared_data = prepare_shared_data(data_area[area], neighbor)
                receive_shared_data!(data_area[neighbor], deepcopy(shared_data), area)
            end
        end

        # calculate mismatches and update convergence flags
        Threads.@threads for area in areas_id
            dopf_method.update_method(data_area[area])
        end

        print_level = 1
                # print solution
        print_iteration(data_area, print_level, [info])

        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        iteration += 1

    end

    return data_area, alpha_trace 
end

case_path = "data/case14.m"
data = parse_file(case_path)
data_area, alpha_trace = quick_adaptive_test(data)

model_type = ACPPowerModel
optimizer = Ipopt.Optimizer 
compare_solution(data, data_area, model_type, optimizer)