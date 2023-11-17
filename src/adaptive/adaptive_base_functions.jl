using PowerModelsADA 
using Suppressor
using LinearAlgebra
include("util.jl")

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

function change_alpha(data, self_key, tau_inc, tau_dec)
    alpha_max = data["parameter"]["alpha_max"]
    alpha_min = data["parameter"]["alpha_min"]
    mu_inc = data["parameter"]["mu_inc"]
    mu_dec = data["parameter"]["mu_dec"]
    k1 = first(keys(data["alpha"]))
    alpha_pq = data["alpha"][k1]["pf"][first(keys(data["alpha"][k1]["pf"]))]
    alpha_vt = data["alpha"][k1]["vm"][first(keys(data["alpha"][k1]["vm"]))]
    if data["mismatch"][self_key] > mu_inc * data["dual_residual"][self_key]
        alpha_pq = alpha_pq * ( 1 + tau_inc)
        alpha_vt = alpha_vt * (1 + tau_inc)
    elseif data["dual_residual"][self_key] > mu_dec * data["mismatch"][self_key]
        alpha_pq = alpha_pq / ( 1 + tau_dec)
        alpha_vt = alpha_vt / (1 + tau_dec)
    end
    if alpha_pq > alpha_max 
        alpha_pq = alpha_max 
    elseif alpha_pq < alpha_min 
        alpha_pq = alpha_min 
    end
    if alpha_vt > alpha_max 
        alpha_vt = alpha_max 
    elseif alpha_vt < alpha_min 
        alpha_vt = alpha_min 
    end
    return alpha_pq, alpha_vt 
end

function initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol, baseline_alpha_pq, baseline_alpha_vt)
    areas_id = get_areas_id(data)

    if length(areas_id) < 2
        error("Number of areas is less than 2, at least 2 areas is needed")
    end

    # decompose the system into subsystems
    data_area = Dict{Int64, Any}()
    for area in areas_id
        data_area[area] = decompose_system(data, area)
    end

    # initilize distributed power model parameters
    for area in areas_id
        dopf_method.initialize_method(data_area[area], model_type, max_iteration=max_iteration, termination_measure="mismatch_dual_residual", mismatch_method="norm", tol=tol, tol_dual=du_tol, save_data=["solution", "shared_variable", "received_variable", "mismatch"])
        assign_alpha!(data_area[area],baseline_alpha_pq,baseline_alpha_vt)
    end

    return data_area 
end

function run_some_iterations(data_area::Dict{Int,Any}, dopf_method::Module, model_type::DataType, optimizer, iteration::Int, tau_inc::AbstractFloat, tau_dec::AbstractFloat, iters_to_run::Int, rng)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    reward_residual_data = Dict("primal" => [], "dual" => []) 
    agent_residual_data = Dict(i => Dict("primal" => [], "dual" => []) for i in areas_id)
    stop_iteration = iteration + iters_to_run 
    conv_iter = -1
    first_time = true  
    while iteration < stop_iteration 

        info = @capture_out begin
            for area in areas_id
                result = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
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
    
        alpha_copy = Dict(area => deepcopy(data_area[area]["alpha"]) for area in areas_id)
        # calculate mismatches and update convergence flags
        Threads.@threads for area in areas_id
            dopf_method.update_method(data_area[area])
            save_solution!(data_area[area])
        end
        for area in areas_id 
            data_area[area]["alpha"] = alpha_copy[area]
        end

        # solve local problem and update solution
        for area in areas_id 
            data = data_area[area] 
            alpha_pq, alpha_vt = change_alpha(data, string(area), tau_inc, tau_dec)
            assign_alpha!(data_area[area], alpha_pq, alpha_vt)
        end        

        #Record the residuals 
        for area in areas_id
            push!(agent_residual_data[area]["dual"], data_area[area]["dual_residual"][string(area)])
            push!(agent_residual_data[area]["primal"], data_area[area]["mismatch"][string(area)])
        end
        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))
    
        print_level = 1
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        if flag_convergence
            if first_time
                conv_iter = iteration 
                first_time = false 
            end
        end
        if mod(iteration,5) == 0 
            print_iteration(data_area, print_level, [info])
            println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
            println()
        end
        iteration += 1
    end

    report_area_id = rand(rng, areas_id)
    println("conv iter: ", conv_iter)
    return reward_residual_data, agent_residual_data[report_area_id], data_area, iteration, flag_convergence, conv_iter 
end

function run_to_end(data_area::Dict{Int,Any}, tau_inc, tau_dec;
         dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000)    
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    reward_residual_data = Dict("primal" => [], "dual" => []) 

    while iteration < max_iteration && !flag_convergence
        # solve local problem and update solution     
        info = @capture_out begin
            for area in areas_id
                result = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
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
    
        alpha_copy = Dict(area => deepcopy(data_area[area]["alpha"]) for area in areas_id)
        # calculate mismatches and update convergence flags
        Threads.@threads for area in areas_id
            dopf_method.update_method(data_area[area])
            save_solution!(data_area[area])
        end
        for area in areas_id 
            data_area[area]["alpha"] = alpha_copy[area]
        end

        # solve local problem and update solution
        for area in areas_id 
            data = data_area[area] 
            alpha_pq, alpha_vt = change_alpha(data, string(area), tau_inc, tau_dec)
            if mod(iteration, 5) == 0
                println(area, " ", alpha_pq, " ", alpha_vt)
                assign_alpha!(data_area[area], alpha_pq, alpha_vt)
            end
        end        

        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))
    
        print_level = 1

        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)

        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println(flag_convergence)
        println()
        iteration += 1
    end

    return data_area 
end

function run_to_end(data_area::Dict{Int,Any}, Q, tau_inc_action_set, tau_dec_action_set, n_history, update_alpha_freq;
        dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)  
    alphas = Dict(i => Dict("pq" => 0, "vt" => 0) for i in areas_id)
    agent_residual_data = Dict(i => Dict("primal" => [], "dual" => []) for i in areas_id)
    iteration = 1 
    reward_residual_data = Dict("primal" => [], "dual" => []) 
    state_trace = Dict(i => [] for i in areas_id)
    while iteration < max_iteration && !flag_convergence

        # solve local problem and update solution
        info = @capture_out begin
            for area in areas_id
                result = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
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
    
        alpha_copy = Dict(area => deepcopy(data_area[area]["alpha"]) for area in areas_id)
        # calculate mismatches and update convergence flags
        Threads.@threads for area in areas_id
            dopf_method.update_method(data_area[area])
            save_solution!(data_area[area])
        end
        for area in areas_id 
            data_area[area]["alpha"] = alpha_copy[area]
        end

        if mod(iteration-n_history-1,update_alpha_freq) == 0 && iteration >= n_history 
            for area in areas_id 
                state = vcat(sigmoid_norm_primal(agent_residual_data[area]["primal"][end-n_history+1:end]),sigmoid_norm_dual(agent_residual_data[area]["dual"][end-n_history+1:end]))
                push!(state_trace[area],deepcopy(state))
                a = argmax(Q(state))
                println(Q(state))
                n_actions_tau_dec = length(tau_dec_action_set)
                tau_inc_idx = Int(ceil(a/n_actions_tau_dec))
                tau_dec_idx = a - (tau_inc_idx-1)*n_actions_tau_dec 
                tau_inc = tau_inc_action_set[tau_inc_idx]
                tau_dec = tau_dec_action_set[tau_dec_idx]
                data = data_area[area] 
                alpha_pq, alpha_vt = change_alpha(data, string(area), tau_inc, tau_dec)
                assign_alpha!(data_area[area], alpha_pq, alpha_vt)
                println("a: ", a, " tau_inc: ", tau_inc, " tau_dec: ", tau_dec)
                println("pq: ", alpha_pq)
                println("vt: ", alpha_vt)
            end
        end

        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))

        #Record the residuals 
        for area in areas_id
            push!(agent_residual_data[area]["dual"], data_area[area]["dual_residual"][string(area)])
            push!(agent_residual_data[area]["primal"], data_area[area]["mismatch"][string(area)])
        end        
    
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)

        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println()
        iteration += 1
    end

    return data_area, state_trace 
end

function quick_adaptive_test(data;
    dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000, tol=1e-4) 
    
    areas_id = get_areas_id(data)
    # get areas ids
    ## decompose the system into subsystems
    data_area = decompose_system(data)

    # initilize distributed power model parameters
    for area in areas_id
        dopf_method.initialize_method(data_area[area], model_type; max_iteration=max_iteration, tol=tol)
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
                result = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
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