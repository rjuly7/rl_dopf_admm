using Pkg
Pkg.activate(".")

using PowerModelsADA 
using Ipopt 
include("dopf_solve_steps.jl")

casename = "case14"
data = parse_file("data/$casename.m")
n_areas = 3
need_csv = 1
if (need_csv == 1)
    partition_path= "data/$casename"*"_$n_areas.csv"
    assign_area!(data, partition_path)
end

model_type = ACPPowerModel
dopf_method = admm_2lvl_methods 
optimizer = Ipopt.Optimizer 

data_area =  solve_admm_2lvl_iterations(data, model_type, optimizer, dopf_method; print_level=1, multiprocessors=false, tol=1e-4, tol_dual=1e-2, alpha=800, termination_measure="mismatch_dual_residual")
#data_area = solve_dopf(data, ACPPowerModel, Ipopt.Optimizer, admm_methods; print_level=1, multiprocessors=false, tol=1e-4, tol_inner=1e-4)

println(compare_solution(data, data_area, ACPPowerModel, Ipopt.Optimizer))


# arrange and get areas id
arrange_areas_id!(data)
areas_id = get_areas_id(data)
diameter = get_diameter(data)

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
    dopf_method.initialize_method(data_area[area], model_type, print_level=1, tol=1e-4, tol_dual=1e-2, alpha=800, termination_measure="mismatch_dual_residual")
end

# get global parameters
max_iteration = 1000

# initialize the algorithms global counters
iteration = 1
flag_convergence = false

# start iteration
while iteration <= max_iteration && !flag_convergence

    # solve local problem and update solution
    info = @capture_out begin
        Threads.@threads for area in areas_id
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
        dopf_method.update_method1(data_area[area])
    end

    # print solution
    print_iteration(data_area, print_level, [info])
    if mod(iteration, 10) == 1 && print_level == 1
        #print_iteration(data_area, print_level, [info])
        pri_resid = sqrt(sum(data_area[area]["mismatch"][string(area)]^2 for area in areas_id))
        du_resid = sqrt(sum(data_area[area]["dual_residual"][string(area)]^2 for area in areas_id))
        println("Iteration: ", iteration, "   pri: ", pri_resid, "  du: ", du_resid)
    end

    # check global convergence and update iteration counters
    flag_convergence = update_global_flag_convergence(data_area)
    iteration += 1

end