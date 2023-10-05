using PowerModelsADA 
using Suppressor
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


function initialize_dopf(data, model_type, dopf_method, max_iteration, tol, initial_alpha)
    areas_id = get_areas_id(data)

    if length(areas_id) < 2
        error("Number of areas is less than 2, at least 2 areas is needed")
    end

    # decompose the system into subsystems
    data_area = Dict{Int64, Any}()
    for area in areas_id
        data_area[area] = decompose_system(data, area)
    end

    # get areas ids
    areas_id = get_areas_id(data_area)

    # initilize distributed power model parameters
    for area in areas_id
        dopf_method.initialize_method(data_area[area], model_type, alpha=initial_alpha, tol=tol, max_iteration=max_iteration, termination_measure="mismatch_dual_residual", save_data=["solution", "shared_variable", "received_variable", "mismatch"])
    end

    return data_area 
end

function run_some_iterations(data_area::Dict{Int,Any}, dopf_method::Module, model_type::DataType, optimizer, iteration::Int, adaptive_params, iters_to_run::Int, rng)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    for area in areas_id 
        data_area[area]["parameter"]["eta_inc"] = adaptive_params[1]
        data_area[area]["parameter"]["eta_dec"] = adaptive_params[2]
        data_area[area]["parameter"]["mu_inc"] = adaptive_params[3]
        data_area[area]["parameter"]["mu_dec"] = adaptive_params[4]
    end
    reward_residual_data = Dict("primal" => [], "dual" => []) 
    agent_residual_data = Dict(i => Dict("primal" => [], "dual" => []) for i in areas_id)
    stop_iteration = iteration + iters_to_run 
    while iteration < stop_iteration   
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
            save_solution!(data_area[area])
        end

        #Record the residuals 
        for area in areas_id
            push!(agent_residual_data[area]["dual"], data_area[area]["dual_residual"][string(area)])
            push!(agent_residual_data[area]["primal"], data_area[area]["mismatch"][string(area)])
        end
        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))
    
        print_level = 1
        # print solution
        print("Converged: ", flag_convergence, "   ")
        print_iteration(data_area, print_level, [info])
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)

        iteration += 1
    end

    report_area_id = rand(rng, areas_id)
    return reward_residual_data, agent_residual_data[report_area_id], data_area, iteration, flag_convergence #(flag_convergence && dual_convergence)
end



function run_to_end(data_area::Dict{Int,Any}, adaptive_params;
         dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000)    
    flag_convergence = false
    areas_id = get_areas_id(data_area)

    for area in areas_id 
        data_area[area]["parameter"]["eta_inc"] = adaptive_params[1]
        data_area[area]["parameter"]["eta_dec"] = adaptive_params[2]
        data_area[area]["parameter"]["mu_inc"] = adaptive_params[3]
        data_area[area]["parameter"]["mu_dec"] = adaptive_params[4]
    end
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    iteration = 1 
    while iteration < max_iteration && !flag_convergence
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
            save_solution!(data_area[area])
        end

        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)

        iteration += 1
    end

    return data_area 
end

function run_to_end(data_area::Dict{Int,Any}, Q, action_set, baselines, n_history, update_alpha_freq;
        dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)  

    for area in areas_id 
        data_area[area]["parameter"]["eta_inc"] = baselines[1]
        data_area[area]["parameter"]["eta_dec"] = baselines[2]
        data_area[area]["parameter"]["mu_inc"] = baselines[3]
        data_area[area]["parameter"]["mu_dec"] = baselines[4]
    end
    agent_residual_data = Dict(i => Dict("primal" => [], "dual" => []) for i in areas_id)
    iteration = 1 
    while iteration < max_iteration && !flag_convergence

        if mod(iteration-n_history-1,update_alpha_freq) == 0 && iteration >= n_history 
            for area in areas_id 
                state = vcat(agent_residual_data[area]["primal"][end-n_history+1:end],agent_residual_data[area]["dual"][end-n_history+1:end])
                a = argmax(Q(state))
                println(Q(state))
                n_actions = Dict(i => length(action_set[i]) for i in eachindex(action_set))
                action_idcs = get_action_idcs(a,n_actions)
                adaptive_params = [action_set[i][action_idcs[i]] for i in eachindex(action_set)]
                for area in areas_id 
                    data_area[area]["parameter"]["eta_inc"] = adaptive_params[1]
                    data_area[area]["parameter"]["eta_dec"] = adaptive_params[2]
                    data_area[area]["parameter"]["mu_inc"] = adaptive_params[3]
                    data_area[area]["parameter"]["mu_dec"] = adaptive_params[4]
                end
            end
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
            save_solution!(data_area[area])
        end

        #Record the residuals 
        for area in areas_id
            push!(agent_residual_data[area]["dual"], data_area[area]["dual_residual"][string(area)])
            push!(agent_residual_data[area]["primal"], data_area[area]["mismatch"][string(area)])
        end        
    
        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)

        iteration += 1
    end

    return data_area 
end

function quick_adaptive_test(data, dual_measure=false;
    dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000, tol=1e-4) 
    
    areas_id = get_areas_id(data)
    # get areas ids
    ## decompose the system into subsystems
    data_area = decompose_system(data)

    # initilize distributed power model parameters
    for area in areas_id
        dopf_method.initialize_method(data_area[area], model_type; max_iteration=max_iteration, tol=tol)
        #dopf_method.initialize_method(data_area[area], model_type; termination_measure= "mismatch_dual_residual", max_iteration=max_iteration, tol=tol)
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