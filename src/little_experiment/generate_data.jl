casename = "case118_3" 
n_areas = 3
need_csv = 0
N = 2 
run_num = "1"

if lastindex(ARGS) >= 5
    casename = ARGS[1]
    n_areas = parse(Int,ARGS[2])
    need_csv = parse(Int, ARGS[3])
    N = parse(Int, ARGS[4])
    run_num = ARGS[5]
else
    println("Using default, no command line arguments given")
end

using Distributed 
addprocs(n_areas)

@everywhere using Pkg 
@everywhere Pkg.activate(".")

using LinearAlgebra
@everywhere using PowerModelsADA 
@everywhere using Ipopt 
using BSON 
@everywhere using PowerModels 
using JuMP 
@everywhere using Random 
@everywhere include("data_collection.jl")
Random.seed!(123)
case_path = "data/$casename.m"
data = parse_file(case_path)
if (need_csv == 1)
    partition_path= "data/$casename"*"_$n_areas.csv"
    assign_area!(data, partition_path)
end
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1000

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

primal_map, dual_map,load_map = create_maps(data,data_area)
bson("data/little_experiment/$casename"*"_maps.bson", Dict("primal" => primal_map, "dual" => dual_map, "load" => load_map))

pq_bounds = [100,3200]
vt_bounds = [1400,7000]
pq_lower = pq_bounds[1]
pq_upper = pq_bounds[2]
vt_lower = vt_bounds[1]
vt_upper = vt_bounds[2]

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

lower_bounds,upper_bounds = get_bounds(pq_bounds,vt_bounds,data_area)
alpha_config,alpha_vector = get_hyperparameter_configuration(deepcopy(data_area),pq_bounds,vt_bounds) #pull initial config 

stuff = BSON.load("data/hyperband/linucb_$casename"*"_$pq_lower"*"_$pq_upper"*"_$vt_lower"*"_$vt_upper"*"_1.jl")
trace_params = stuff["trace"]
V = trace_params["V"][end]
y = trace_params["y"][end]
hat_theta = inv(V)*y 
areas_id = get_areas_id(data_area)
nv = sum(1 for area in areas_id for n in keys(data_area[area]["shared_variable"]) for v in keys(data_area[area]["shared_variable"][n]) for k in keys(data_area[area]["shared_variable"][n][v]))

model = Model(Ipopt.Optimizer)
@variable(model, lower_bounds[i] <= a[i=1:nv] <= upper_bounds[i])
@objective(model, Max, dot(hat_theta,a))
optimize!(model)
println(termination_status(model))

alpha_vector = value.(a)
alpha_config = vector_to_config(alpha_vector,deepcopy(data_area))
initial_iters = 20

all_vectors = @distributed (append!) for i = 1:N 
    data_area,load_vector = get_perturbed_data_area(data,model_type,dopf_method,tol,du_tol,max_iteration,load_map)
    vars_vector = run_dopf_mp(initial_config,alpha_config,data_area,model_type,dopf_method,primal_map,dual_map,initial_iters)
    [(load_vector,vars_vector)]
end

X = all_vectors[1][1]
y = all_vectors[1][2]
for ds in eachindex(all_vectors)[2:end] 
    global X = hcat(X,all_vectors[ds][1])
    global y = hcat(y,all_vectors[ds][2])
end

bson("data/little_experiment/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))

