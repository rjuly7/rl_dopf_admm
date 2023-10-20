using PowerModelsADA 
using Suppressor

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


function initialize_dopf(data, model_type, dopf_method, max_iteration, tol)
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
        dopf_method.initialize_method(data_area[area], model_type, tol=tol, max_iteration=max_iteration, termination_measure="mismatch_dual_residual", save_data=["solution", "shared_variable", "received_variable", "mismatch"])
    end

    return data_area 
end

function run_some_iterations(data_area::Dict{Int,Any}, dopf_method::Module, model_type::DataType, optimizer, iteration::Int, alpha_pq::Int, alpha_vt::Int, iters_to_run::Int, rng)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    reward_residual_data = Dict("primal" => [], "dual" => []) 
    agent_residual_data = Dict(i => Dict("primal" => [], "dual" => []) for i in areas_id)
    stop_iteration = iteration + iters_to_run 
    my_flag_convergence = false 
    eabs = 1e-5
    erel = 1e-4 
    while iteration < stop_iteration 
        # solve local problem and update solution
        for area in areas_id 
            assign_alpha!(data_area[area], alpha_pq, alpha_vt)
        end        
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
        
        cur_pri_norm = LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]],2)
        cur_du_norm = LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]],2)
        num_shared = sum([1 for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]])
        num_dual = sum([1 for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]])
        if reward_residual_data["primal"][end] <= sqrt(num_shared)*eabs + erel*cur_pri_norm
            if reward_residual_data["dual"][end] <= sqrt(num_dual)*eabs + erel*cur_du_norm
                my_flag_convergence = true 
            end
        end
    
        print_level = 1
        # print solution
        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println("epri: ", sqrt(num_shared)*eabs + erel*cur_pri_norm, "  edu: ", sqrt(num_dual)*eabs + erel*cur_du_norm)
        println()
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        print("Converged: ", my_flag_convergence, "   ")

        iteration += 1
    end

    report_area_id = rand(rng, areas_id)
    return reward_residual_data, agent_residual_data[report_area_id], data_area, iteration, my_flag_convergence #(flag_convergence && dual_convergence)
end



function run_to_end(data_area::Dict{Int,Any}, alpha_pq, alpha_vt;
         dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000)    
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    my_flag_convergence = false 
    eabs = 1e-5 
    erel = 1e-4 
    reward_residual_data = Dict("primal" => [], "dual" => []) 

    while iteration < max_iteration && !my_flag_convergence #!flag_convergence
        # solve local problem and update solution
        for area in areas_id 
            assign_alpha!(data_area[area], alpha_pq, alpha_vt)
        end        
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

        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))

        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        cur_pri_norm = LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]],2)
        cur_du_norm = LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]],2)
        num_shared = sum([1 for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]])
        num_dual = sum([1 for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]])
        if reward_residual_data["primal"][end] <= sqrt(num_shared)*eabs + erel*cur_pri_norm
            if reward_residual_data["dual"][end] <= sqrt(num_dual)*eabs + erel*cur_du_norm
                my_flag_convergence = true 
            end
        end
        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println("epri: ", sqrt(num_shared)*eabs + erel*cur_pri_norm, "  edu: ", sqrt(num_dual)*eabs + erel*cur_du_norm)
        println(my_flag_convergence)
        println()
        iteration += 1
    end

    return data_area 
end

function run_to_end(data_area::Dict{Int,Any}, Q, pq_action_set, vt_action_set, alpha_pq, alpha_vt, n_history, update_alpha_freq;
        dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)  
    alphas = Dict(i => Dict("pq" => alpha_pq, "vt" => alpha_vt) for i in areas_id)
    agent_residual_data = Dict(i => Dict("primal" => [], "dual" => []) for i in areas_id)
    iteration = 1 
    eabs = 1e-5 
    erel = 1e-4 
    my_flag_convergence = false 
    reward_residual_data = Dict("primal" => [], "dual" => []) 

    while iteration < max_iteration && !my_flag_convergence #!flag_convergence

        if mod(iteration-n_history-1,update_alpha_freq) == 0 && iteration >= n_history 
            for area in areas_id 
                state = vcat(agent_residual_data[area]["primal"][end-n_history+1:end],agent_residual_data[area]["dual"][end-n_history+1:end])
                a = argmax(Q(state))
                n_actions_vt = length(vt_action_set)
                pq_idx = Int(ceil(a/n_actions_vt))
                vt_idx = a - (pq_idx-1)*n_actions_vt 
                alphas[area]["pq"] = pq_action_set[pq_idx]
                alphas[area]["vt"] = vt_action_set[vt_idx]
                println("pq: ", alphas[area]["pq"])
                println("vt: ", alphas[area]["vt"])
            end
        end

        for area in areas_id
            assign_alpha!(data_area[area], alphas[area]["pq"], alphas[area]["vt"])
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

        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))

        #Record the residuals 
        for area in areas_id
            push!(agent_residual_data[area]["dual"], data_area[area]["dual_residual"][string(area)])
            push!(agent_residual_data[area]["primal"], data_area[area]["mismatch"][string(area)])
        end        
    
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        cur_pri_norm = LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]],2)
        cur_du_norm = LinearAlgebra.norm([value for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]],2)
        num_shared = sum([1 for area in areas_id for i in keys(data_area[area]["shared_variable"]) for j in keys(data_area[area]["shared_variable"][i]) for (k,value) in data_area[area]["shared_variable"][i][j]])
        num_dual = sum([1 for area in areas_id for i in keys(data_area[area]["dual_variable"]) for j in keys(data_area[area]["dual_variable"][i]) for (k,value) in data_area[area]["dual_variable"][i][j]])
        if reward_residual_data["primal"][end] <= sqrt(num_shared)*eabs + erel*cur_pri_norm
            if reward_residual_data["dual"][end] <= sqrt(num_dual)*eabs + erel*cur_du_norm
                my_flag_convergence = true 
            end
        end
        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println("epri: ", sqrt(num_shared)*eabs + erel*cur_pri_norm, "  edu: ", sqrt(num_dual)*eabs + erel*cur_du_norm)
        println()
        iteration += 1
    end

    return data_area 
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