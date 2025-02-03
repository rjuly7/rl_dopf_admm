function get_alpha_configs(linucb_agents)
    configs = Dict()
    for n in keys(linucb_agents)
        V = linucb_agents[n]["V"]
        inv_V = inv(V)
        y = linucb_agents[n]["y"]
        hat_theta = inv_V*y 
        lower_bounds = linucb_agents[n]["lower_bounds"]
        upper_bounds = linucb_agents[n]["upper_bounds"]

        nv = length(linucb_agents[n]["upper_bounds"])
        T = 500 
        beta = 1 + sqrt(2*log(T)+nv*log((nv+T)/nv))
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "print_level", 0)
        @variable(model, lower_bounds[i]/1000 <= a[i=1:nv] <= upper_bounds[i]/1000)
        #@variable(model, u)
        #@objective(model, Max, dot(hat_theta,a) + sqrt(beta)*u)
        #@constraint(model, transpose(a)*inv_V*a == u^2)
        @objective(model, Max, dot(hat_theta,a))
        optimize!(model)
        println(termination_status(model))
        
        alpha_vector = 1000*value.(a)
        alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
        configs[n] = alpha_config
    end
    return configs 
end

function run_with_configs(data_area::Dict{Int64, <:Any},configs,optimizer,linucb_agents,N)
    dopf_method = adaptive_admm_methods
    model_type = ACPPowerModel
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    max_iteration = first(data_area)[2]["option"]["max_iteration"]

    region_bounds = linucb_agents[N]["region_bounds"]
    reward = -1000
    reached_bound = [false for ii=1:N] 
    reached_bound_iter = [0 for ii=1:N]
    bound_region = 1

    #data_area = perturb_loads(data_area)

    while iteration < max_iteration && !(reached_bound[end])
        # overwrite any changes the adaptive algorithm made to alphas in last iteration 
        for area in areas_id 
            #fill in with alphas from linucb_agent in charge of current residual region 
            data_area[area]["alpha"] = deepcopy(linucb_agents[bound_region]["alpha_config"][area])
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
        for ii=1:N
            if reached_bound[ii] == false 
                this_bound = region_bounds[ii]
                if pri_resid <= this_bound[1] && du_resid <= this_bound[2]
                    reached_bound_iter[ii] = iteration 
                    reached_bound[ii] = true 
                    bound_region = ii+1
                end
            end
        end

        if mod(iteration, 2) == 1
            #print_iteration(data_area, print_level, [info])
            println("Agent : $N  bound_region $bound_region Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    return reached_bound_iter,iteration   
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