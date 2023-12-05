using Ipopt 
using BSON 
using Plots 
using PowerModelsADA 

casename = "case30"
n_areas = 3
need_csv = 1
case_path = "data/$casename.m"
data = parse_file(case_path)
if (need_csv == 1)
    partition_path= "data/$casename"*"_$n_areas.csv"
    assign_area!(data, partition_path)
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

function run_then_return_val_loss_mp(data_area::Dict{Int64, <:Any},alpha_config,optimizer,model_type,dopf_method)
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 

    #data_area = perturb_loads(data_area)

    while iteration < max_iteration && !flag_convergence
        # overwrite any changes the adaptive algorithm made to alphas in last iteration 

        for area in areas_id 
            data_area[area]["alpha"] = deepcopy(alpha_config[area])
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

        print_level = 1
        # print solution
        if mod(iteration, 10) == 1
            #print_iteration(data_area, print_level, [info])
            pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
            du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
            println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
        end

        iteration += 1
    end

    return iteration 
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

model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1000
optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

alpha_pq = 400
alpha_vt = 4000 
alpha_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

n_iter = run_then_return_val_loss_mp(data_area,alpha_config,optimizer,model_type,dopf_method)