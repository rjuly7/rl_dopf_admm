using Pkg 
Pkg.activate(".")

using StatsBase 
using PowerModelsADA
using Ipopt 
using JSON 
using BSON 
using Suppressor 
include("adaptive_base_functions.jl")

function run_to_end_test(data_area::Dict{Int,Any}, tau_inc, tau_dec;
    dopf_method=adaptive_admm_methods, model_type=ACPPowerModel, optimizer=Ipopt.Optimizer, max_iteration=1000)    
    flag_convergence = false
    #We want to store the 2-norm of primal and dual residuals
    #at each iteration
    areas_id = get_areas_id(data_area)
    iteration = 1 
    reward_residual_data = Dict("primal" => [], "dual" => []) 

    while iteration < max_iteration && !flag_convergence
        # solve local problem and update solution     
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

        alpha_copy = Dict(area => deepcopy(data_area[area]["alpha"]) for area in areas_id)
        # calculate mismatches and update convergence flags
        Threads.@threads for area in areas_id
            dopf_method.update_method(data_area[area])
            save_solution!(data_area[area])
        end
        for area in areas_id 
            data_area[area]["alpha"] = alpha_copy[area]
        end

        # solve local problem and update solution
        for area in areas_id 
            if iteration > 20 
                data = data_area[area] 
                alpha_pq, alpha_vt = change_alpha(data, string(area), tau_inc, tau_dec)
                assign_alpha!(data_area[area], alpha_pq, alpha_vt)
            end
            if mod(iteration, 5) == 0
                k1 = first(keys(data_area[1]["alpha"]))
                alpha_pq = data_area[1]["alpha"][k1]["pf"][first(keys(data_area[1]["alpha"][k1]["pf"]))]
                alpha_vt = data_area[1]["alpha"][k1]["vm"][first(keys(data_area[1]["alpha"][k1]["vm"]))]
                println("CHECKING pq: ", alpha_pq, "  vt: ", alpha_vt)
            end
        end        

        push!(reward_residual_data["dual"], sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id)))
        push!(reward_residual_data["primal"], sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id)))

        print_level = 1

        # check global convergence and update iteration counters
        flag_convergence = update_global_flag_convergence(data_area)

        print_level = 1
        # print solution
        print_iteration(data_area, print_level, [info])
        println("pri: ", reward_residual_data["primal"][end], "  du: ", reward_residual_data["dual"][end])
        println(flag_convergence)
        println()
        iteration += 1
    end

    return data_area 
end

baseline_alpha_pq = 400 
baseline_alpha_vt = 4000 
max_iteration = 1000
case_path = "data/case118_3.m"
data = parse_file(case_path)
model_type = ACPPowerModel
tol = 1e-4 
du_tol = 0.1 
tau_inc = 0.01
tau_dec = 0.01 

data_area = initialize_dopf(data, model_type, adaptive_admm_methods, max_iteration, tol, du_tol, baseline_alpha_pq, baseline_alpha_vt)
data_area = run_to_end_test(data_area, tau_inc, tau_dec)

