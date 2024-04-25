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

function run_then_return_val_loss(data_area,alpha_config,initial_config,optimizer,initial_iters)
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

# "update the area data solution dictionary"
# function update_solution!(pm::AbstractPowerModel, solution::Dict{String, <:Any})
#     solution["solution"] = pm.data["solution"]
#     for variable in keys(solution["solution"])
#         for idx in keys(solution["solution"][variable])
#             solution["solution"][variable][idx] = JuMP.value(PowerModelsADA._var(pm, variable, idx))
#         end
#     end
# end

# ""
# function solve_pmada_model(data::Dict{String,<:Any}, model_type::Type, optimizer, build_method;
#         ref_extensions=[], solution_processors=[], relax_integrality=false,
#         multinetwork=false, multiconductor=false, kwargs...)

#     if multinetwork != _IM.ismultinetwork(data)
#         model_requirement = multinetwork ? "multi-network" : "single-network"
#         data_type = _IM.ismultinetwork(data) ? "multi-network" : "single-network"
#     end

#     if multiconductor != ismulticonductor(data)
#         model_requirement = multiconductor ? "multi-conductor" : "single-conductor"
#         data_type = ismulticonductor(data) ? "multi-conductor" : "single-conductor"
#     end

#     pm = instantiate_pmada_model(data, model_type, build_method; ref_extensions=ref_extensions, kwargs...)

#     result = optimize_model!(pm, relax_integrality=relax_integrality, optimizer=optimizer, solution_processors=solution_processors)

#     return result
# end

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
                    rewards[n] = - iteration 
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


function run_then_return_val_loss_mp(data_area::Dict{Int64, <:Any},alpha_config,initial_config,optimizer,initial_iters)
    dopf_method = adaptive_admm_methods
    model_type = ACPPowerModel
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration

    # lookup dictionaries for worker-area pairs
    areas_id = get_areas_id(data_area)
    worker_id = Distributed.workers()
    number_workers = length(worker_id)

    k = 1
    area_worker = Dict()
    for i in areas_id
        if k > number_workers
            k = 1
        end
        area_worker[i] = worker_id[k]
        k += 1 
    end
    worker_area = Dict([i => findall(x -> x==i, area_worker) for i in worker_id if i in values(area_worker)])

    # initiate communication channels 
    comms = Dict(0 => Dict(area => Distributed.RemoteChannel(1) for area in areas_id))
    for area1 in areas_id
        comms[area1] = Dict()
        for area2 in [0; areas_id]
            if area1 != area2 
                comms[area1][area2] = Distributed.RemoteChannel(area_worker[area1])
            end
        end
    end

    # initilize distributed power model parameters
    for area in areas_id
        put!(comms[0][area], data_area[area])
    end

    # get global parameters
    max_iteration = first(data_area)[2]["option"]["max_iteration"]

    # initialize the algorithms global counters
    iteration = 1
    global_flag_convergence = false
    global_counters = Dict{Int64, Any}()

    # share global variables
    Distributed.@everywhere keys(worker_area) begin
        comms = $comms
        areas_id = $areas_id
        worker_area = $worker_area
        area_worker = $area_worker
        dopf_method = $dopf_method
        model_type = $model_type
        optimizer = $optimizer
        area_id = worker_area[myid()]
        data_local = Dict{Int64, Any}(area => take!(comms[0][area]) for area in area_id)
    end

    # start iteration
    while iteration <= max_iteration && !global_flag_convergence

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

        Distributed.@everywhere keys(worker_area) begin
            for area in area_id
                # solve local problem and update solution
                result = solve_pmada_model(data_local[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
                update_data!(data_local[area], result["solution"])
        
                # send data to neighboring areas
                for neighbor in data_local[area]["neighbors"] 
                    shared_data = prepare_shared_data(data_local[area], neighbor)
                    put!(comms[area][neighbor], shared_data)
                end
            end
        end

        Distributed.@everywhere keys(worker_area) begin
            for area in area_id
                # receive data to neighboring areas
                for neighbor in data_local[area]["neighbors"] 
                    received_data = take!(comms[neighbor][area])
                    receive_shared_data!(data_local[area], received_data, neighbor)
                end

                # calculate and share mismatches
                dopf_method.update_method(data_local[area])
                counters = Dict("option"=> data_local[area]["option"], "counter" => data_local[area]["counter"], "mismatch" => data_local[area]["mismatch"])
                if data_local[area]["option"]["termination_measure"] in ["dual_residual", "mismatch_dual_residual"]
                    counters["dual_residual"] = data_local[area]["dual_residual"]
                end
                put!(comms[area][0], deepcopy(counters))
            end
        end

        # receive the mismatches from areas
        for area in areas_id
            counters = take!(comms[area][0])
            global_counters[area] = counters
        end

        # print progress
        if mod(iteration, 10) == 1
            #print_iteration(data_area, print_level, [info])
            pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
            du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end    

        # update flag convergence and iteration number
        global_flag_convergence = update_global_flag_convergence(global_counters)
        iteration += 1
    end

    # receive the final solution
    Distributed.@everywhere keys(worker_area) begin
        for area in area_id
            # send the area data
            put!(comms[area][0], data_local[area])
        end
    end

    for area in areas_id
        data_area[area] = take!(comms[area][0])
    end

    # close the communication channels
    for i in keys(comms)
        for j in keys(comms[i])
            close(comms[i][j])
        end
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
    set_optimizer_attribute(model, "print_level", 0)
    #set_optimizer_attribute(model, "NonConvex", 2)
    @variable(model, lower_bounds[i]/1000 <= a[i=1:nv] <= upper_bounds[i]/1000)
    @variable(model, u)
    @objective(model, Max, dot(hat_theta,a) + sqrt(beta)*u)
    @constraint(model, transpose(a)*inv_V*a == u^2)
    optimize!(model)
    println(termination_status(model))

    return 1000*value.(a),V,y 
end

function get_perturbed_data_area(data)
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

function initialize_lin_ucb(pq_bounds,vt_bounds,region_bounds,data_area, lambda)
    alpha_config,alpha_vector = get_hyperparameter_configuration(deepcopy(data_area),pq_bounds,vt_bounds) #pull initial config 
    nv = length(alpha_vector)
    V = lambda*Matrix(1.0I, nv, nv)
    y = zeros(nv,1)
    lower_bounds,upper_bounds = get_bounds(pq_bounds,vt_bounds,data_area)
    reward = 0
    trace_params = Dict("V" => [V], "y" => [y], "a" => [alpha_vector], "reward" => [])
    beta = 1 + sqrt(2*log(T)+nv*log((nv+T)/nv))
    linucb_agent = Dict("nv" => nv, "V" => V, "y" => y, "lower_bounds" => lower_bounds, "upper_bounds" => upper_bounds, "reward" => reward, "trace_params" => trace_params,
                            "beta" => beta, "alpha_config" => alpha_config, "alpha_vector" => alpha_vector, "region_bounds" => region_bounds)
    return linucb_agent
end

function run_linucb(T,data_area,data,pq_bounds,vt_bounds,optimizer,lambda;
                        region_bounds=[(0.01,10),(0.001,1),(1e-4,0.1)])
    linucb_agents = Dict{Int,Any}()
    for n in eachindex(region_bounds)
        linucb_agents[n] = initialize_lin_ucb(pq_bounds, vt_bounds, region_bounds, data_area, lambda)
    end

    for i=1:T 
        data_area = get_perturbed_data_area(deepcopy(data))
        rewards = run_then_return_val_loss_sp(data_area,linucb_agents,optimizer)
        for n=1:n_agents 
            reward = rewards[n]
            push!(linucb_agents[n]["trace_params"]["reward"], reward)
            la = linucb_agents[n]
            alpha_vector,V,y = LinUCB(la["V"],la["alpha_vector"],la["reward"],la["y"],la["beta"],la["nv"],la["lower_bounds"],la["upper_bounds"])
            linucb_agents[n]["alpha_vector"] = alpha_vector 
            linucb_agents[n]["V"] = V 
            linucb_agents[n]["y"] = y 
            alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
            linucb_agents[n]["alpha_config"] = alpha_config 
            push!(linucb_agents[n]["trace_params"]["V"], V)
            push!(linucb_agents[n]["trace_params"]["y"], y)
            push!(linucb_agents[n]["trace_params"]["a"], alpha_vector)
        end
    end
        
    return linucb_agents  
end

function LinUCB_quad(V,action,reward,y,beta,nv,lower_bounds,upper_bounds,Vq,yq)
    reward = reward/50 
    action = action/1000 
    action_m = reshape(action, length(action), 1)
    V = V + action_m*transpose(action_m)
    y = y + reward*action_m 
    inv_V = inv(V)
    aq = vcat(action_m, [i^2 for i in action_m])
    Vq = Vq + aq*transpose(aq)
    yq = yq + reward*aq 
    hat_kappa = inv(Vq)*yq 
    #hat_theta = inv_V * y

    model = Model(Ipopt.Optimizer)
    #set_optimizer_attribute(model, "NonConvex", 2)
    @variable(model, lower_bounds[i]/1000 <= a[i=1:nv] <= upper_bounds[i]/1000)
    @variable(model, u)
    @objective(model, Max, sum([(hat_kappa[i]*a[i] + hat_kappa[i+nv]*a[i]^2) for i=1:nv]) + sqrt(beta)*u)
    @constraint(model, transpose(a)*inv_V*a == u^2)
    optimize!(model)
    println(termination_status(model))

    return 1000*value.(a),V,y 
end

function run_linucb_quad(T,data_area,pq_bounds,vt_bounds,initial_config,optimizer,lambda)
    alpha_config,alpha_vector = get_hyperparameter_configuration(deepcopy(data_area),pq_bounds,vt_bounds) #pull initial config 
    nv = length(alpha_vector)
    V = lambda*Matrix(1.0I, nv, nv)
    y = zeros(nv,1)
    Vq = lambda*Matrix(1.0I, 2*nv, 2*nv)
    yq = zeros(2*nv, 1) 
    lower_bounds,upper_bounds = get_bounds(pq_bounds,vt_bounds,data_area)
    reward = 0
    trace_params = Dict("V" => [V], "y" => [y], "a" => [alpha_vector], "reward" => [], "Vq" => [], "yq" => [])
    beta = 1 + sqrt(2*log(T)+nv*log((nv+T)/nv))

    for i=1:T 
        reward = run_then_return_val_loss(deepcopy(data_area),alpha_config,initial_config,optimizer)
        push!(trace_params["reward"], reward)
        println(i, " :", reward)
        alpha_vector,V,y = LinUCB_quad(V,alpha_vector,reward,y,beta,nv,lower_bounds,upper_bounds,Vq,yq)
        alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
        push!(trace_params["V"], V)
        push!(trace_params["y"], y)
        push!(trace_params["a"], alpha_vector)
        push!(trace_params["Vq"], Vq)
        push!(trace_params["yq"], yq)
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