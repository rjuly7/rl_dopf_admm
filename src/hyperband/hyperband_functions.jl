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
        dopf_method.initialize_method(data_area[area], model_type, max_iteration=max_iteration, termination_measure="mismatch_dual_residual", mismatch_method="norm", tol=tol, tol_dual=du_tol, save_data=["solution", "shared_variable", "received_variable", "mismatch"])
    end

    return data_area 
end

#n i.i.d samples from some distribution defined over the hyperparameter configuration space 
function get_hyperparameter_configuration(n,data_area,pq_bounds,vt_bounds)
    alpha_configs = [Dict(area_id => deepcopy(data_area[area_id]["alpha"]) for area_id in keys(data_area)) for i in 1:n]
    for n_idx=1:n
        for area_id in keys(data_area)
            data = data_area[area_id]
            for neighbor in keys(data["alpha"])
                for opf_var in keys(data["alpha"][neighbor])
                    for i in keys(data["alpha"][neighbor][opf_var])
                        if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                            alpha_configs[n_idx][area_id][neighbor][opf_var][i] = rand()*(pq_bounds[2]-pq_bounds[1])+pq_bounds[1] 
                        else
                            alpha_configs[n_idx][area_id][neighbor][opf_var][i] = rand()*(vt_bounds[2]-vt_bounds[1])+vt_bounds[1] 
                        end
                    end
                end
            end
        end
    end
    return alpha_configs 
end

#setting intial alpha values 
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

function perturb_loads(data_area)
    for area in keys(data_area)
        for (i,load) in data_area[area]["load"]
            r = rand() - 0.5
            data_area[area]["load"][i]["pd"] = load["pd"] + load["pd"] * r
            data_area[area]["load"][i]["qd"] = load["qd"] + load["qd"] * r
        end
    end
    return data_area 
end

function run_then_return_val_loss(data_area,alpha_config,initial_config,optimizer)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 

    data_area = perturb_loads(data_area)

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
        if mod(iteration, 5) == 1
            print_iteration(data_area, print_level, [info])
            pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
            du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    return iteration 
end

function maxk(a, k)
    b = partialsortperm(a, 1:k, rev=true)
    return collect(zip(b, a[b]))
end

function top_k(configs,losses,k)
    top_idcs_vals = maxk(losses, k)
    return [configs[top_idcs_vals[i][1]] for i in eachindex(top_idcs_vals)], top_idcs_vals
end

function get_run_count(eta,R)
    smax = Int(floor(log(eta,R)))
    B = (smax + 1)*R 
    run_count = 0
    for s = smax:-1:0
        n = Int(ceil(B/R*eta^s/(s+1)))
        r = R/(eta^s)
        println("s: ", s, "  n: ", n, "  r: ", r)
        for i = 0:s
            n_i = Int(floor(n/(eta^i)))
            r_i = r*eta^i 
            println("s: ", s, "  n: ", n, " i: ", i, " n_i: ", n_i, " r_i: ", r_i)
            for iii = 1:n
                for rr=1:r_i 
                    run_count += 1
                end
            end
            println(run_count)
        end
        println("Total for s: ",run_count)
    end 
    return run_count 
end 