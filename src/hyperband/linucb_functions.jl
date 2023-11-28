using Suppressor
#n i.i.d samples from some distribution defined over the hyperparameter configuration space 
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

function run_then_return_val_loss(data_area,alpha_config,initial_config,optimizer)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 

    #data_area = perturb_loads(data_area)

    while iteration < max_iteration && !flag_convergence
        # overwrite any changes the adaptive algorithm made to alphas in last iteration 
        if iteration >= 20
            for area in areas_id 
                data_area[area]["alpha"] = deepcopy(alpha_config[area])
            end
        else
            for area in areas_id 
                data_area[area]["alpha"] = deepcopy(initial_config[area])
            end
        end
            

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
            save_solution!(data_area[area])
        end

        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)

        print_level = 1
        # print solution
        if mod(iteration, 10) == 1
            print_iteration(data_area, print_level, [info])
            pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
            du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    return 200 - iteration 
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
        dopf_method.initialize_method(data_area[area], model_type, max_iteration=max_iteration, termination_measure="mismatch_dual_residual", mismatch_method="norm", tol=tol, tol_dual=du_tol, save_data=["solution", "shared_variable", "received_variable", "mismatch"])
    end

    return data_area 
end

function LinUCB(V,action,reward,y,beta,nv,lower_bounds,upper_bounds)
    reward = reward/50 
    action = action/1000 
    a = reshape(action, length(action), 1)
    V = V + a*transpose(a)
    y = y + reward*a 
    inv_V = inv(V)
    hat_theta = inv_V * y

    model = Model(Ipopt.Optimizer)
    #set_optimizer_attribute(model, "NonConvex", 2)
    @variable(model, lower_bounds[i] <= a[i=1:nv] <= upper_bounds[i])
    @variable(model, u)
    @objective(model, Max, dot(hat_theta,a) + sqrt(beta)*u)
    @constraint(model, transpose(a)*inv_V*a == u^2)
    optimize!(model)
    println(termination_status(model))

    return value.(a),V,y 
end

function run_linucb(T,data_area,pq_bounds,vt_bounds,initial_config,optimizer,lambda)
    alpha_config,alpha_vector = get_hyperparameter_configuration(deepcopy(data_area),pq_bounds,vt_bounds) #pull initial config 
    nv = length(alpha_vector)
    V = lambda*Matrix(1.0I, nv, nv)
    y = zeros(nv,1)
    lower_bounds,upper_bounds = get_bounds(pq_bounds,vt_bounds,data_area)
    reward = 0
    trace_params = Dict("V" => [V], "y" => [y], "a" => [alpha_vector], "reward" => [])
    beta = 1 + sqrt(2*log(T)+nv*log((nv+T)/nv))

    for i=1:T 
        reward = run_then_return_val_loss(deepcopy(data_area),alpha_config,initial_config,optimizer)
        push!(trace_params["reward"], reward)
        println(i, " :", reward)
        alpha_vector,V,y = LinUCB(V,alpha_vector,reward,y,beta,nv,lower_bounds,upper_bounds)
        alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
        push!(trace_params["V"], V)
        push!(trace_params["y"], y)
        push!(trace_params["a"], alpha_vector)
    end
        
    return reward,alpha_config,trace_params  
end

function LinUCB_discrete(V,a_idx,reward,y,beta,nv)
    reward = reward/50
    a = zeros(nv,1)
    a[a_idx] = 1
    V = V + a*transpose(a)
    y = y + reward*a 
    inv_V = inv(V)
    hat_theta = inv_V * y

    val_list = []
    for a_idx=1:nv 
        a_vec = zeros(nv,1)
        a_vec[a_idx] = 1         
        push!(val_list, dot(a_vec,hat_theta) + sqrt(beta)*sqrt(transpose(a_vec)*inv_V*a_vec)[1])
    end

    best_val,best_idx = findmax(val_list)
    println("Best val: ", best_val, "  best arm: ", best_idx)

    return best_idx,V,y 
end

function run_linucb_discrete(T,data_area,pqs,vts,initial_config,optimizer,lambda)
    pq_vt_vec = []
    a_idx_vec = []
    #Generate all combinations 
    counter = 1
    for p in pqs 
        for v in vts 
            push!(pq_vt_vec,(p,v))
            push!(a_idx_vec,counter)
        end
    end

    nv = length(a_idx_vec)
    a_idx = rand(1:nv)
    alpha_config = set_hyperparameter_configuration(deepcopy(data_area),pq_vt_vec[a_idx][1],pq_vt_vec[a_idx][2]) #pull initial config 

    V = lambda*Matrix(1.0I, nv, nv)
    y = zeros(nv,1)
    reward = 0
    trace_params = Dict("V" => [V], "y" => [y], "a" => [a_idx], "reward" => [])
    beta = 1 + sqrt(2*log(T)+nv*log((nv+T)/nv))

    for i=1:T 
        reward = run_then_return_val_loss(deepcopy(data_area),alpha_config,initial_config,optimizer)
        push!(trace_params["reward"], reward)
        println(i, " :", reward)
        a_idx,V,y = LinUCB_discrete(V,a_idx,reward,y,beta,nv)
        alpha_config = idx_to_config(a_idx,pq_vt_vec,deepcopy(data_area))
        push!(trace_params["V"], V)
        push!(trace_params["y"], y)
        push!(trace_params["a"], a_idx)
    end
        
    return reward,alpha_config,trace_params  
end