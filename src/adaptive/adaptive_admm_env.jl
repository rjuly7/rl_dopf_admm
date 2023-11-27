include("adaptive_base_functions.jl")
include("util.jl")
using IntervalSets
using Ipopt 

struct AdaptiveADMMEnvParams
    tol::AbstractFloat
    du_tol::AbstractFloat 
    max_iteration::Int 
    n_history::Int 
    baseline_alpha_pq::Int 
    baseline_alpha_vt::Int 
    baseline_tau_inc::AbstractFloat 
    baseline_tau_dec::AbstractFloat 
    model_type::DataType
    optimizer 
    tau_update_freq::Int 
end

function doAdaptiveADMMEnvParams(
    tol,
    du_tol,
    max_iteration,
    n_history,
    baseline_alpha_pq,
    baseline_alpha_vt,
    baseline_tau_inc,
    baseline_tau_dec,
    model_type,
    optimizer,
    tau_update_freq
)
    return AdaptiveADMMEnvParams(
        tol,
        du_tol,
        max_iteration,
        n_history,
        baseline_alpha_pq,
        baseline_alpha_vt,
        baseline_tau_inc,
        baseline_tau_dec,
        model_type,
        optimizer,
        tau_update_freq
    )
end 

mutable struct AdaptiveADMMEnv <: AbstractEnv
    params::AdaptiveADMMEnvParams
    action_space::Base.OneTo{Int}
    tau_inc_action_set::Vector{AbstractFloat}
    tau_dec_action_set::Vector{AbstractFloat}
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

function AdaptiveADMMEnv(data, tau_inc_action_set, tau_dec_action_set, rng;
    model_type = ACPPowerModel,
    tol = 1e-4,
    du_tol=0.1,
    max_iteration = 1000,
    n_history = 20,
    baseline_alpha_pq = 400,
    baseline_alpha_vt = 4000,
    baseline_tau_inc = 0,
    baseline_tau_dec = 0,
    optimizer = Ipopt.Optimizer,
    tau_update_freq = 10
)
    params = doAdaptiveADMMEnvParams(tol, du_tol, max_iteration, n_history, baseline_alpha_pq, baseline_alpha_vt, baseline_tau_inc, baseline_tau_dec, model_type, optimizer, tau_update_freq)
    total_actions = length(tau_inc_action_set)*length(tau_dec_action_set)
    action_space = Base.OneTo(total_actions) 
    state_space = Vector{ClosedInterval{AbstractFloat}}() 
    for i = 1:2*n_history
        push!(state_space,ClosedInterval(0.0,1.0))
    end
    tau_a = rand(action_space)
    data_area = initialize_dopf(data, model_type, adaptive_admm_methods, max_iteration, tol, du_tol, baseline_alpha_pq, baseline_alpha_vt)
    env = AdaptiveADMMEnv(
        params,
        action_space,
        tau_inc_action_set,
        tau_dec_action_set,
        Space(state_space),
        ones(Float64, 2*n_history),
        tau_a,
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

RLBase.action_space(env::AdaptiveADMMEnv) = env.action_space
RLBase.state_space(env::AdaptiveADMMEnv) = env.observation_space
RLBase.reward(env::AdaptiveADMMEnv) = env.reward 
RLBase.is_terminated(env::AdaptiveADMMEnv) = env.done
RLBase.state(env::AdaptiveADMMEnv) = env.state

function RLBase.reset!(env::AdaptiveADMMEnv) 
    println("Resetting!!!")
    println()
    println()
    env.iteration = 1
    env.data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt)
    pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged, conv_iter = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), env.params.baseline_tau_inc, env.params.baseline_tau_dec, env.params.n_history, rng)
    env.data_area = pol_data_area 
    env.iteration = pol_iteration 
    env.state = vcat(sigmoid_norm_primal(agent_pol_residual_data["primal"]),sigmoid_norm_dual(agent_pol_residual_data["dual"]))
    env.residual_history_buffer["primal"] = agent_pol_residual_data["primal"]
    env.residual_history_buffer["dual"] = agent_pol_residual_data["dual"]
    env.done = false
    nothing
end

function (env::AdaptiveADMMEnv)(a::Int)
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

function _step!(env::AdaptiveADMMEnv, a)
    n_actions_tau_dec = length(env.tau_dec_action_set)
    tau_inc_idx = Int(ceil(a/n_actions_tau_dec))
    tau_dec_idx = a - (tau_inc_idx-1)*n_actions_tau_dec 
    tau_inc = env.tau_inc_action_set[tau_inc_idx]
    tau_dec = env.tau_dec_action_set[tau_dec_idx]
    save_data_area = deepcopy(env.data_area)
    save_iteration = deepcopy(env.iteration)
    println("Running base:")
    base_residual_data, agent_base_residual_data, base_data_area, base_iteration, base_converged, base_conv_iter = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), env.params.baseline_tau_inc, env.params.baseline_tau_dec, env.params.tau_update_freq, rng)
    println("Running pol:")
    pol_residual_data, agent_pol_residual_data, pol_data_area, pol_iteration, pol_converged, pol_conv_iter = run_some_iterations(deepcopy(env.data_area), adaptive_admm_methods, env.params.model_type, env.params.optimizer, copy(env.iteration), tau_inc, tau_dec, env.params.tau_update_freq, rng)
    #Compute reward 
    Rb = (base_residual_data["primal"][end] - pol_residual_data["primal"][end])/base_residual_data["primal"][end] + (base_residual_data["dual"][end] - pol_residual_data["dual"][end])/base_residual_data["dual"][end]
    println(a, " ", tau_inc, " ", tau_dec)
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
    env.residual_history_buffer["primal"], primal_state = update_residual_history(env.residual_history_buffer["primal"], agent_pol_residual_data["primal"], copy(env.iteration)-1, env.params.n_history, env.params.tau_update_freq)
    env.residual_history_buffer["dual"], dual_state = update_residual_history(env.residual_history_buffer["dual"], agent_pol_residual_data["dual"], copy(env.iteration)-1, env.params.n_history, env.params.tau_update_freq)
    env.state = vcat(sigmoid_norm_primal(primal_state),sigmoid_norm_dual(dual_state))
    nothing
end

function test_baseline()
    data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt)
    data_area = run_to_end(data_area, env.params.baseline_tau_inc, env.params.baseline_tau_dec)
    return data_area 
end

function test_policy(Q)
    data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt)
    data_area, state_trace = run_to_end(data_area, Q, env.tau_inc_action_set, env.tau_dec_action_set, env.params.n_history, env.params.tau_update_freq)
    return data_area, state_trace 
end