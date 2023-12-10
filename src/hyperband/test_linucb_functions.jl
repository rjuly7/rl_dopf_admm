function get_iterations(data_area,alpha_config,initial_config,optimizer,initial_iters)
    dopf_method = adaptive_admm_methods
    model_type = ACPPowerModel
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    max_iteration = 1000
    #data_area = perturb_loads(data_area)

    while iteration < max_iteration && !flag_convergence
        # overwrite any changes the adaptive algorithm made to alphas in last iteration 
        if iteration >= initial_iters 
            for area in areas_id 
                data_area[area]["alpha"] = deepcopy(alpha_config[area])
            end
        else
            for area in areas_id 
                data_area[area]["alpha"] = deepcopy(initial_config[area])
            end
        end
            

        #info = @capture_out begin
        for area in areas_id
            result = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
            update_data!(data_area[area], result["solution"])
        end
        #end

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
        flag_convergence = update_global_flag_convergence(data_area)

        print_level = 1
        # print solution
        if mod(iteration, 20) == 1
            print_iteration(data_area, print_level, [info])
            pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
            du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    return iteration 
end

function perturb_loads(data)
    model_type = ACPPowerModel
    dopf_method = adaptive_admm_methods 
    tol = 1e-4 
    du_tol = 0.1 
    max_iteration = 1000
    data_orig = deepcopy(data)
    for (i,load) in data_orig["load"]
        r = rand() - 0.4
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