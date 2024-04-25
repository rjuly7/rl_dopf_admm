function run_then_return_val_loss_sp(data_area::Dict{Int64, <:Any},linucb_agents,optimizer)
    dopf_method = adaptive_admm_methods
    model_type = ACPPowerModel
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    max_iteration = first(data_area)[2]["option"]["max_iteration"]

    region_bounds = first(linucb_agents)[2]["region_bounds"]
    flag_bounds = Dict(n => false for n in keys(linucb_agents))
    rewards = Dict(n => 0 for n in keys(linucb_agents))
    flag_keys = sort!(collect(keys(flag_bounds)))

    #data_area = perturb_loads(data_area)

    while iteration < max_iteration && !flag_bounds[flag_keys[end]]
        # overwrite any changes the adaptive algorithm made to alphas in last iteration 
        if iteration == 1 
            alpha_config = linucb_agents[1]["alpha_config"]
            for area in areas_id 
                data_area[area]["alpha"] = deepcopy(alpha_config[area])
            end
        else
            false_flags = [i for i in keys(flag_bounds) if flag_bounds[i] == false]
            sort!(false_flags)
            n_agent = false_flags[1]
            alpha_config = linucb_agents[n_agent]["alpha_config"]
            for area in areas_id  
                data_area[area]["alpha"] = deepcopy(alpha_config[area])
            end
        end

        for area in areas_id 
            dr = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
            update_data!(data_area[area], dr["solution"])        
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

        # check global convergence and update iteration counters
        #flag_convergence = update_global_flag_convergence(data_area)
        pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
        du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
        for n in keys(flag_bounds)
            if flag_bounds[n] == false 
                if pri_resid <= region_bounds[n][1] && du_resid <= region_bounds[n][2]
                    flag_bounds[n] = true 
                    if n == 1
                        rewards[n] = - iteration 
                    else
                        rewards[n] = - (iteration + rewards[n-1])
                    end 
                    println("Flag bounds $n reward ", rewards[n])
                end
            end
        end

        if mod(iteration, 10) == 1
            #print_iteration(data_area, print_level, [info])
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    return rewards 
end

function perturb_loads(data_orig)
    model_type = ACPPowerModel
    dopf_method = adaptive_admm_methods 
    tol = 1e-4 
    du_tol = 0.1 
    max_iteration = 1000
    data = deepcopy(data_orig)
    for (i,load) in data_orig["load"]
        r = rand() - 0.5
        data["load"][i]["pd"] = load["pd"]*(1+r)
        data["load"][i]["qd"] = load["qd"]*(1+r)
    end
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    checker = solve_ac_opf(data,optimizer)
    println(checker["termination_status"] == LOCALLY_SOLVED)
    while checker["termination_status"] != LOCALLY_SOLVED
        for (i,load) in data_orig["load"]
            r = rand() - 0.4
            data["load"][i]["pd"] = load["pd"]*(1+r)
            data["load"][i]["qd"] = load["qd"]*(1+r)
        end
        checker = solve_ac_opf(data,optimizer)
    end
    data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)
    return data_area 
end