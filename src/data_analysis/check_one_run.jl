using Pkg 
cd("C:/Users/User/Documents/rl_dopf_admm/")
Pkg.activate(".")

using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using StatsBase 
using PowerModelsADA
using Ipopt 
using JSON 
using BSON 
using Plots 
include("C:/Users/User/Documents/rl_dopf_admm/src/original/admm_env.jl")
include("C:/Users/User/Documents/rl_dopf_admm/src/original/base_functions.jl")

case_path = "data/case118_3.m"
data = parse_file(case_path)

rng = StableRNG(123)

pq_alpha_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
vt_alpha_values = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5500, 7000]

env = ADMMEnv(data, pq_alpha_values, vt_alpha_values, rng, baseline_alpha_pq = 400, baseline_alpha_vt = 4000, alpha_update_freq = 10)
ns, na = length(state(env)), length(action_space(env))

run_num = 12
sub_num = 24000

ns, na = length(state(env)), length(action_space(env))

agent = BSON.load("data/saved_agents/agent_$run_num"*"_$sub_num.bson")["agent"]

#Compare to baseline 
#base_data_area = test_baseline()
#baseline_iterations = base_data_area[1]["counter"]["iteration"]
Qt = agent.policy.learner.target_approximator
data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol)
polt_data_area, statet_trace, alphat_trace = run_to_end(data_area, Qt, env.pq_action_set, env.vt_action_set, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, env.params.alpha_update_freq)

Q = agent.policy.learner.approximator 
data_area = initialize_dopf(env.data, env.params.model_type, adaptive_admm_methods, env.params.max_iteration, env.params.tol, env.params.du_tol)
pol_data_area, state_trace, alpha_trace = run_to_end(data_area, Q, env.pq_action_set, env.vt_action_set, env.params.baseline_alpha_pq, env.params.baseline_alpha_vt, env.params.n_history, env.params.alpha_update_freq)
policyt_iterations = polt_data_area[1]["counter"]["iteration"]
policy_iterations = pol_data_area[1]["counter"]["iteration"]
println("policy with target: ", policyt_iterations, "  policy not target: ", policy_iterations)

ep_rewards = [r for r in agent.trajectory.traces.reward if r > 10]
plot(ep_rewards)