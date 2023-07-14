using PowerModelsADA 
using Suppressor

function initialize_dopf(data, model_type, dopf_method, alpha, max_iteration, tol)
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
        dopf_method.initialize_method(data_area[area], model_type, tol=tol, alpha=alpha, max_iteration=max_iteration, termination_measure="mismatch_dual_residual", save_data=["solution", "shared_variable", "received_variable", "mismatch"])
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
    for area in areas_id
        for neighbor in keys(data_area[area]["alpha"])
            for opf_var in keys(data_area[area]["alpha"][neighbor])
                for i in keys(data_area[area]["alpha"][neighbor][opf_var])
                    if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                        data_area[area]["alpha"][neighbor][opf_var][i] = alpha_pq 
                    else
                        data_area[area]["alpha"][neighbor][opf_var][i] = alpha_vt
                    end
                end
            end
        end
    end
    stop_iteration = iteration + iters_to_run 
    while iteration < stop_iteration 
        # solve local problem and update solution
        info = @capture_out begin
            for area in areas_id
                result = solve_model(data_area[area], model_type, optimizer, dopf_method.build_method, solution_processors=dopf_method.post_processors)
                update_data!(data_area[area], result["solution"])
            end
        end

        #Record the residuals 
        primal_residuals = Dict()
        for area in areas_id
            calc_dual_residual!(data_area[area])
            primal_shared = data_area[area]["shared_variable"]
            primal_residuals[area] = sqrt(sum((primal_shared[i][j][k] - (data_area[area]["received_variable"][i][j][k] + primal_shared[i][j][k])/2)^2 for i in keys(primal_shared) for j in keys(primal_shared[i]) for k in keys(primal_shared[i][j]))) 
        end 
        for area in areas_id
            push!(agent_residual_data[area]["dual"], data_area[area]["dual_residual"][string(area)])
            push!(agent_residual_data[area]["primal"], primal_residuals[area])
        end
        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(primal_residuals[area]^2 for area in areas_id)))
    
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
        print("Converged: ", flag_convergence, "   ")
        print_iteration(data_area, print_level, [info])
        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)
        #global_dual_residual = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
        # if global_dual_residual < tol
        #     dual_convergence = true 
        # end
        iteration += 1
    end

    report_area_id = rand(rng, areas_id)
    return reward_residual_data, agent_residual_data[report_area_id], data_area, iteration, flag_convergence #(flag_convergence && dual_convergence)
end

