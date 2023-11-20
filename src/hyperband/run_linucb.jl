using LinearAlgebra
using JuMP 
using Gurobi 

function LinUCB(V,action,reward,y,beta,nv,lower_bounds,upper_bounds):
    a = reshape(action, length(action), 1)
    V = V + a*transpose(a)
    y = y + reward*a 
    inv_V = np.linalg.inv(V)
    hat_theta = inv_V * y

    model = Model(Gurobi.Optimizer)
    @variable(model, lower_bounds[i] <= a[i=1:nv] <= upper_bounds[i])
    @variable(model, y)
    @objective(model, Max, dot(hat_theta,a) + sqrt(beta)*y)
    @constraint(model, transpose(a)*inv_V*a = y^2)

    return np.argmax(ucb),V,y

function run_linucb(data_area,pq_bounds,vt_bounds,initial_config,optimizer)
    alpha_configs,alpha_vector = get_hyperparameter_configuration(data_area,pq_bounds,vt_bounds) #pull initial config 
    nv = length(alpha_vector)
    V = lambda*Matrix(1.0I, nv, nv)

end

case_path = "data/case118_3.m"
data = parse_file(case_path)
model_type = ACPPowerModel
dopf_method = adaptive_admm_methods 
tol = 1e-4 
du_tol = 0.1 
max_iteration = 1000
optimizer = Ipopt.Optimizer 

data_area = initialize_dopf(data, model_type, dopf_method, max_iteration, tol, du_tol)

pq_bounds = [150,800]
vt_bounds = [3000,5000]

alpha_pq = 400
alpha_vt = 4000 
initial_config = set_hyperparameter_configuration(data_area,alpha_pq,alpha_vt)

R = 50
eta = 3 
best_configs = run_hyperband_perturb_loads(R,eta,data_area,pq_bounds,vt_bounds,initial_config,optimizer)
bson("data/hyperband/linucb_$run_num.jl", Dict("best_configs" => best_configs))

