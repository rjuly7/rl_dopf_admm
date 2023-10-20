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
using LinearAlgebra

function quick_adaptive_test(data, alpha_pq, alpha_vt;
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

    for area in areas_id 
        data_area[area]["parameter"]["eta_inc"] = 0.01
        data_area[area]["parameter"]["eta_dec"] = 0.01
        data_area[area]["parameter"]["mu_inc"] = 1.5
        data_area[area]["parameter"]["mu_dec"] = 1.5
    end
    
    # initialize the algorithms global counters
    iteration = 1
    flag_convergence = false
    my_flag_convergence = false 
    reward_residual_data = Dict("primal" => [], "dual" => []) 
    pri_norm = []
    du_norm = []
    eabs = 1e-5
    erel = 1e-4 
    alpha_trace = Dict(i => [] for i in areas_id)
    for area in areas_id 
        assign_alpha!(data_area[area], alpha_pq, alpha_vt)
    end       
    # start iteration
    while iteration < max_iteration && !my_flag_convergence

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

        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))

        push!(pri_norm, LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]],2))
        push!(du_norm, LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]],2))
        num_shared = sum([1 for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]])
        num_dual = sum([1 for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]])
        if reward_residual_data["primal"][end] <= sqrt(num_shared)*eabs + erel*pri_norm[end]
            if reward_residual_data["dual"][end] <= sqrt(num_dual)*eabs + erel*du_norm[end]
                my_flag_convergence = true 
            end
        end

        print_level = 1
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println("epri: ", sqrt(num_shared)*eabs + erel*pri_norm[end], "  edu: ", sqrt(num_dual)*eabs + erel*du_norm[end])
        println()

        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        iteration += 1

    end

    return data_area, alpha_trace, pri_norm, du_norm 
end

function assign_alpha!(data, alpha_pq, alpha_vt)
    for neighbor in keys(data["alpha"])
        for opf_var in keys(data["alpha"][neighbor])
            for i in keys(data["alpha"][neighbor][opf_var])
                if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                    data["alpha"][neighbor][opf_var][i] = alpha_pq 
                else
                    data["alpha"][neighbor][opf_var][i] = alpha_vt
                end
            end
        end
    end
end

function quick_constant_test(data, alpha_pq, alpha_vt;
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
        for area in areas_id 
            assign_alpha!(data_area[area], alpha_pq, alpha_vt)
        end        
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

function quick_constant_test(data, alpha_pq, alpha_vt;
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
    my_flag_convergence = false 
    alpha_trace = Dict(i => [] for i in areas_id)
    reward_residual_data = Dict("primal" => [], "dual" => []) 
    pri_norm = []
    du_norm = []
    eabs = 1e-5
    erel = 1e-4 
    # start iteration
    while iteration < max_iteration && !my_flag_convergence
        for area in areas_id 
            assign_alpha!(data_area[area], alpha_pq, alpha_vt)
        end        
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

        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))

        push!(pri_norm, LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]],2))
        push!(du_norm, LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]],2))
        num_shared = sum([1 for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]])
        num_dual = sum([1 for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]])
        if reward_residual_data["primal"][end] <= sqrt(num_shared)*eabs + erel*pri_norm[end]
            if reward_residual_data["dual"][end] <= sqrt(num_dual)*eabs + erel*du_norm[end]
                my_flag_convergence = true 
            end
        end

        print_level = 1
                # print solution
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println("epri: ", sqrt(num_shared)*eabs + erel*pri_norm[end], "  edu: ", sqrt(num_dual)*eabs + erel*du_norm[end])
        println()

        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        iteration += 1

    end

    return data_area, alpha_trace, reward_residual_data, pri_norm, du_norm 
end

case_path = "data/case118_3.m"
data = parse_file(case_path)
data_area, alpha_trace, pri_norm, du_norm = quick_adaptive_test(data, 400, 4000)

model_type = ACPPowerModel
optimizer = Ipopt.Optimizer 
println(compare_solution(data, data_area, model_type, optimizer))

data_area, alpha_trace, reward_residual_data, pri_norm, du_norm = quick_constant_test(data, 400, 4000)
println(compare_solution(data, data_area, model_type, optimizer))
#takes 277

