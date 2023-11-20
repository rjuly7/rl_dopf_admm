include("base_functions.jl")
include("util.jl")
using IntervalSets
using Ipopt 

struct ADMMEnvParams
    tol::AbstractFloat
    du_tol::AbstractFloat 
    max_iteration::Int 
    n_history::Int 
    baseline_alpha_pq::Int 
    baseline_alpha_vt::Int 
    model_type::DataType
    optimizer 
    alpha_update_freq::Int 
end

function ADMMEnvParams(
    tol,
    du_tol,
    max_iteration,
    n_history,
    baseline_alpha_pq,
    baseline_alpha_vt,
    model_type,
    optimizer,
    alpha_update_freq
)
    ADMMEnvParams(
        tol,
        du_tol,
        max_iteration,
        n_history,
        baseline_alpha_pq,
        baseline_alpha_vt,
        model_type,
        optimizer,
        alpha_update_freq
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
    residual_history_buffer::Dict{String,Any}
end

function ADMMEnv(data, pq_action_set, vt_action_set, rng;
    model_type = ACPPowerModel,
    tol = 1e-4,
    du_tol=0.1,
    max_iteration = 1000,
    n_history = 20,
    baseline_alpha_pq = 400,
    baseline_alpha_vt = 4000,
    optimizer = Ipopt.Optimizer ,
    alpha_update_freq = 10
)
    params = ADMMEnvParams(tol, du_tol, max_iteration, n_history, baseline_alpha_pq, baseline_alpha_vt, model_type, optimizer, alpha_update_freq)
    total_actions = length(pq_action_set)*length(vt_action_set)
    action_space = Base.OneTo(total_actions) 
    state_space = Vector{ClosedInterval{AbstractFloat}}() 
    for i = 1:2*n_history
        push!(state_space,ClosedInterval(0.0,1.0))
    end
    alpha = rand(action_space)
    data_area = initialize_dopf(data, model_type, adaptive_admm_methods, max_iteration, tol, du_tol)
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
        rng,
        Dict("primal" => [], "dual" => [])
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
    env.data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol)
    pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged, conv_iter = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, rng)
    env.data_area = pol_data_area 
    env.iteration = pol_iteration 
    env.state = vcat(sigmoid_norm_primal(agent_pol_residual_data["primal"]),sigmoid_norm_dual(agent_pol_residual_data["dual"]))
    env.residual_history_buffer["primal"] = agent_pol_residual_data["primal"]
    env.residual_history_buffer["dual"] = agent_pol_residual_data["dual"]
    env.done = false
    nothing
end

function (env::ADMMEnv)(a::Int)
    @assert a in env.action_space
    env.action = a
    _step!(env, a)
end

function update_residual_history(residual_history_buffer, new_residual_data, iteration, n_history, update_freq)
    t = copy(iteration) - update_freq + 1
    for r in new_residual_data
        put_idx = mod(t,n_history)
        if put_idx == 0 
            put_idx = n_history 
        end
        residual_history_buffer[put_idx] = r 
        t += 1 
    end
    start_idx = mod(iteration,n_history)+1 
    state = vcat(residual_history_buffer[start_idx:n_history],residual_history_buffer[1:start_idx-1])
    return residual_history_buffer, state 
end

function _step!(env::ADMMEnv, a)
    n_actions_vt = length(env.vt_action_set)
    pq_idx = Int(ceil(a/n_actions_vt))
    vt_idx = a - (pq_idx-1)*n_actions_vt 
    alpha_pq = env.pq_action_set[pq_idx]
    alpha_vt = env.vt_action_set[vt_idx]
    save_data_area = deepcopy(env.data_area)
    save_iteration = deepcopy(env.iteration)
    base_residual_data, agent_base_residual_data, base_data_area, base_iteration, base_converged, base_conv_iter = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.alpha_update_freq, rng)
    pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged, pol_conv_iter = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), alpha_pq, alpha_vt, env.params.alpha_update_freq, rng)
    #Compute reward 
    Rb = (base_residual_data["primal"][end] - pol_residual_data["primal"][end])/base_residual_data["primal"][end] + (base_residual_data["dual"][end] - pol_residual_data["dual"][end])/base_residual_data["dual"][end]
    println(a, " ", alpha_pq, " ", alpha_vt)
    println((base_residual_data["primal"][end] - pol_residual_data["primal"][end])/base_residual_data["primal"][end], "   ", (base_residual_data["dual"][end] - pol_residual_data["dual"][end])/base_residual_data["dual"][end])
    Rconv = 0 
    if pol_converged
        Rconv = 100 + (200 - pol_conv_iter)
    end
    if (Rb + Rconv) < -100
        env.reward = -100
        println("#########")
        println("Clipping")
        println("##########")
        #env.reward = Rb + Rconv 
    else
        env.reward = Rb + Rconv 
    end
    env.done =
        pol_converged ||
        env.iteration >= env.params.max_iteration 
    env.data_area = pol_data_area 
    env.iteration = pol_iteration 
    env.residual_history_buffer["primal"], primal_state = update_residual_history(env.residual_history_buffer["primal"], agent_pol_residual_data["primal"], copy(env.iteration)-1, env.params.n_history, env.params.alpha_update_freq)
    env.residual_history_buffer["dual"], dual_state = update_residual_history(env.residual_history_buffer["dual"], agent_pol_residual_data["dual"], copy(env.iteration)-1, env.params.n_history, env.params.alpha_update_freq)
    env.state = vcat(sigmoid_norm_primal(primal_state),sigmoid_norm_dual(dual_state))
    nothing
end

function test_baseline()
    data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol)
    data_area = run_to_end(data_area, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt)
    return data_area 
end

function test_policy(Q)
    data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol)
    data_area, state_trace, alpha_trace = run_to_end(data_area, Q, env.pq_action_set, env.vt_action_set, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, env.params.alpha_update_freq)
    return data_area, state_trace, alpha_trace  
end