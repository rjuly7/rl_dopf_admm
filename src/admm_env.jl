include("base_functions.jl")
using IntervalSets
using Ipopt 

struct ADMMEnvParams
    tol::AbstractFloat
    max_iteration::Int 
    n_history::Int 
    baseline_alpha_pq::Int 
    baseline_alpha_vt::Int 
    model_type::DataType
    optimizer 
end

function ADMMEnvParams(
    tol,
    max_iteration,
    n_history,
    baseline_alpha_pq,
    baseline_alpha_vt,
    model_type,
    optimizer 
)
    ADMMEnvParams(
        tol,
        max_iteration,
        n_history,
        baseline_alpha_pq,
        baseline_alpha_vt,
        model_type,
        optimizer 
    )
end 

mutable struct ADMMEnv <: AbstractEnv
    params::ADMMEnvParams
    action_space::Base.OneTo{Int}
    pq_action_set::Vector{Int}
    vt_action_set::Vector{Int}
    observation_space::Space{Vector{ClosedInterval{AbstractFloat}}}
    state::Vector{AbstractFloat}
    action::Int 
    reward::AbstractFloat
    data::Dict{String,Any}
    data_area::Dict{Int,Any}
    done::Bool
    iteration::Int
    rng
end

function ADMMEnv(data, pq_action_set, vt_action_set, rng;
    model_type = ACPPowerModel,
    tol = 1e-4,
    max_iteration = 1000,
    n_history = 20,
    baseline_alpha_pq = 400,
    baseline_alpha_vt = 4000,
    optimizer = Ipopt.Optimizer 
)
    params = ADMMEnvParams(tol, max_iteration, n_history, baseline_alpha_pq, baseline_alpha_vt, model_type, optimizer)
    total_actions = length(pq_action_set)*length(vt_action_set)
    action_space = Base.OneTo(total_actions) 
    state_space = Vector{ClosedInterval{AbstractFloat}}() 
    for i = 1:2*n_history
        push!(state_space,ClosedInterval(0.0,20000.0))
    end
    alpha = rand(action_space)
    data_area = initialize_dopf(data, model_type, admm_methods, max_iteration, tol)
    env = ADMMEnv(
        params,
        action_space,
        pq_action_set,
        vt_action_set,
        Space(state_space),
        ones(Float64, 2*n_history),
        alpha,
        0,
        data,
        data_area,
        false,
        1,
        rng 
    )
    reset!(env)
    env
end

RLBase.action_space(env::ADMMEnv) = env.action_space
RLBase.state_space(env::ADMMEnv) = env.observation_space
RLBase.reward(env::ADMMEnv) = env.reward 
RLBase.is_terminated(env::ADMMEnv) = env.done
RLBase.state(env::ADMMEnv) = env.state

function RLBase.reset!(env::ADMMEnv) 
    println("Resetting!!!")
    println()
    println()
    env.iteration = 1
    env.data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol)
    pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, rng)
    env.data_area = pol_data_area 
    env.iteration = pol_iteration 
    env.state = vcat(agent_pol_residual_data["primal"],agent_pol_residual_data["dual"])
    env.done = false
    nothing
end

function (env::ADMMEnv)(a::Int)
    @assert a in env.action_space
    env.action = a
    _step!(env, a)
end

function _step!(env::ADMMEnv, a)
    n_actions_vt = length(env.vt_action_set)
    pq_idx = Int(ceil(a/n_actions_vt))
    vt_idx = a - (pq_idx-1)*n_actions_vt 
    alpha_pq = env.pq_action_set[pq_idx]
    alpha_vt = env.vt_action_set[vt_idx]
    base_residual_data, agent_base_residual_data, base_data_area, base_iteration, base_converged = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, rng)
    pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), alpha_pq, alpha_vt, env.params.n_history, rng)
    #Compute reward 
    Rb = (base_residual_data["primal"][end] - pol_residual_data["primal"][end])/base_residual_data["primal"][end] + (base_residual_data["dual"][end] - pol_residual_data["dual"][end])/base_residual_data["dual"][end]
    Rconv = 0 
    if pol_converged
        Rconv = 200
    end
    env.reward = Rb + Rconv 
    env.done =
        pol_converged ||
        env.iteration >= env.params.max_iteration 
    env.data_area = pol_data_area 
    env.iteration = pol_iteration 
    env.state = vcat(agent_pol_residual_data["primal"],agent_pol_residual_data["dual"])
    nothing
end

function test_baseline()
    data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol)
    data_area = run_to_end(data_area, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt)
    return data_area 
end

function test_policy(Q)
    data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol)
    data_area = run_to_end(data_area, Q, env.pq_action_set, env.vt_action_set, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history)
    return data_area 
end