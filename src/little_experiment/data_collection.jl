using Suppressor 

function initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)
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
        dopf_method.initialize_method(data_area[area], model_type, max_iteration=max_iteration, termination_measure="mismatch_dual_residual", mismatch_method="norm", tol=tol, tol_dual=du_tol)
    end

    return data_area 
end

function create_maps(data,data_area)
    areas_id = get_areas_id(data_area)
    primal_map = Dict()
    dual_map = Dict()
    load_map = Dict() 
    n_primal = sum(1 for area in areas_id for n in keys(data_area[area]["shared_variable"]) for v in keys(data_area[area]["shared_variable"][n]) for k in keys(data_area[area]["shared_variable"][n][v]))
    counter = 1 
    for area in areas_id 
        primal_map[area] = Dict()
        dual_map[area] = Dict()
        for n in keys(data_area[area]["shared_variable"])
            primal_map[area][n] = Dict()
            dual_map[area][n] = Dict() 
            for v in keys(data_area[area]["shared_variable"][n])
                primal_map[area][n][v] = Dict()
                dual_map[area][n][v] = Dict() 
                for k in keys(data_area[area]["shared_variable"][n][v])
                    primal_map[area][n][v][k] = copy(counter) 
                    dual_map[area][n][v][k] = copy(counter) + n_primal 
                    counter += 1
                end
            end
        end
    end
    counter = 1 
    n_loads = sum(1 for i in keys(data["load"]))
    for (i) in keys(data["load"])
        load_map[i] = Dict()
        load_map[i]["pd"] = counter 
        load_map[i]["qd"] = counter + n_loads 
        counter += 1
    end
    return primal_map, dual_map, load_map 
end

function run_dopf_mp(initial_config,alpha_config,data_area,model_type,dopf_method,primal_map,dual_map,initial_iters)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

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
            dr = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
            update_data!(data_area[area], dr["solution"]) 
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

        # print solution
        if mod(iteration, 10) == 1
            #print_iteration(data_area, print_level, [info])
            pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
            du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    n_shared = sum(1 for area in areas_id for n in keys(data_area[area]["shared_variable"]) for v in keys(data_area[area]["shared_variable"][n]) for k in keys(data_area[area]["shared_variable"][n][v]))

    train_vector = zeros(2*n_shared,1)
    for area in areas_id 
        for n in keys(data_area[area]["shared_variable"])
            for v in keys(data_area[area]["shared_variable"][n])
                for k in keys(data_area[area]["shared_variable"][n][v])
                    train_vector[primal_map[area][n][v][k]] = data_area[area]["shared_variable"][n][v][k] 
                    train_vector[dual_map[area][n][v][k]] = data_area[area]["dual_variable"][n][v][k] 
                end
            end
        end
    end

    return train_vector  
end 

function run_iterations(initial_config,alpha_config,data_area,model_type,dopf_method,primal_map,dual_map,initial_iters)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

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
            dr = solve_pmada_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
            update_data!(data_area[area], dr["solution"]) 
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

        # print solution
        if mod(iteration, 10) == 1
            #print_iteration(data_area, print_level, [info])
            pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
            du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    n_shared = sum(1 for area in areas_id for n in keys(data_area[area]["shared_variable"]) for v in keys(data_area[area]["shared_variable"][n]) for k in keys(data_area[area]["shared_variable"][n][v]))

    train_vector = zeros(2*n_shared,1)
    for area in areas_id 
        for n in keys(data_area[area]["shared_variable"])
            for v in keys(data_area[area]["shared_variable"][n])
                for k in keys(data_area[area]["shared_variable"][n][v])
                    train_vector[primal_map[area][n][v][k]] = data_area[area]["shared_variable"][n][v][k] 
                    train_vector[dual_map[area][n][v][k]] = data_area[area]["dual_variable"][n][v][k] 
                end
            end
        end
    end

    return train_vector, iteration  
end 

function get_hyperparameter_configuration(data_area,pq_bounds,vt_bounds)
    alpha_config = Dict(area_id => deepcopy(data_area[area_id]["alpha"]) for area_id in keys(data_area)) 
    alpha_vector = []
    for area_id in keys(data_area)
        data = data_area[area_id]
        for neighbor in keys(data["alpha"])
            for opf_var in keys(data["alpha"][neighbor])
                for i in keys(data["alpha"][neighbor][opf_var])
                    if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                        aa = rand()*(pq_bounds[2]-pq_bounds[1])+pq_bounds[1] 
                        alpha_config[area_id][neighbor][opf_var][i] = aa 
                        push!(alpha_vector, aa)
                    else
                        aa =  rand()*(vt_bounds[2]-vt_bounds[1])+vt_bounds[1] 
                        alpha_config[area_id][neighbor][opf_var][i] = aa
                        push!(alpha_vector, aa)
                    end
                end
            end
        end
    end
    return alpha_config, alpha_vector 
end

function vector_to_config(alpha_vector,data_area)
    alpha_config = Dict(area_id => deepcopy(data_area[area_id]["alpha"]) for area_id in keys(data_area)) 
    counter = 1
    for area_id in keys(data_area)
        data = data_area[area_id]
        for neighbor in keys(data["alpha"])
            for opf_var in keys(data["alpha"][neighbor])
                for i in keys(data["alpha"][neighbor][opf_var])
                    if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                        alpha_config[area_id][neighbor][opf_var][i] = alpha_vector[counter]
                    else
                        alpha_config[area_id][neighbor][opf_var][i] = alpha_vector[counter]
                    end
                    counter += 1
                end
            end
        end
    end
    return alpha_config 
end


function idx_to_config(a_idx,pq_vt_vec,data_area)
    alpha_config = Dict(area_id => deepcopy(data_area[area_id]["alpha"]) for area_id in keys(data_area)) 
    alpha_pq = pq_vt_vec[a_idx][1]
    alpha_vt = pq_vt_vec[a_idx][2]
    for area_id in keys(data_area)
        data = data_area[area_id]
        for neighbor in keys(data["alpha"])
            for opf_var in keys(data["alpha"][neighbor])
                for i in keys(data["alpha"][neighbor][opf_var])
                    if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                        alpha_config[area_id][neighbor][opf_var][i] = alpha_pq 
                    else
                        alpha_config[area_id][neighbor][opf_var][i] = alpha_vt 
                    end
                end
            end
        end
    end
    return alpha_config 
end

function get_bounds(pq_bounds,vt_bounds,data_area)
    lower_bounds = []
    upper_bounds = []
    for area_id in keys(data_area)
        data = data_area[area_id]
        for neighbor in keys(data["alpha"])
            for opf_var in keys(data["alpha"][neighbor])
                for i in keys(data["alpha"][neighbor][opf_var])
                    if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                        push!(lower_bounds,pq_bounds[1])
                        push!(upper_bounds,pq_bounds[2])
                    else
                        push!(lower_bounds,vt_bounds[1])
                        push!(upper_bounds,vt_bounds[2])
                    end
                end
            end
        end
    end
    return lower_bounds,upper_bounds 
end

function set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)
    alpha_config = Dict(area_id => deepcopy(data_area[area_id]["alpha"]) for area_id in keys(data_area))
    for area_id in keys(data_area)
        data = data_area[area_id]
        for neighbor in keys(data["alpha"])
            for opf_var in keys(data["alpha"][neighbor])
                for i in keys(data["alpha"][neighbor][opf_var])
                    if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                        alpha_config[area_id][neighbor][opf_var][i] = alpha_pq
                    else
                        alpha_config[area_id][neighbor][opf_var][i] = alpha_vt
                    end
                end
            end
        end
    end
    return alpha_config
end

function get_perturbed_data_area(data,model_type,dopf_method,tol,du_tol,max_iteration,load_map)
    data_orig = deepcopy(data)
    n_loads = sum(1 for i in keys(data["load"]))
    load_vector = zeros(2*n_loads,1)
    for (i,load) in data_orig["load"]
        r = rand() - 0.4
        data["load"][i]["pd"] = load["pd"]*(1+r)
        data["load"][i]["qd"] = load["qd"]*(1+r)
        load_vector[load_map[i]["pd"]] = data["load"][i]["pd"]
        load_vector[load_map[i]["qd"]] = data["load"][i]["qd"]
    end
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    checker = solve_ac_opf(data,optimizer)
    println(checker["termination_status"] == LOCALLY_SOLVED)
    while checker["termination_status"] != LOCALLY_SOLVED
        for (i,load) in data_orig["load"]
            r = rand() - 0.4
            data["load"][i]["pd"] = load["pd"]*(1+r)
            data["load"][i]["qd"] = load["qd"]*(1+r)
            load_vector[load_map[i]["pd"]] = data["load"][i]["pd"]
            load_vector[load_map[i]["qd"]] = data["load"][i]["qd"]
        end
        checker = solve_ac_opf(data,optimizer)
        println(checker["termination_status"] == LOCALLY_SOLVED)
    end
    data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)
    return data_area, load_vector 
end