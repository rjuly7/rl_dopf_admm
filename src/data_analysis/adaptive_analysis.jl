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
include("C:/Users/User/Documents/rl_dopf_admm/src/adaptive/adaptive_admm_env.jl")
#include("C:/Users/User/Documents/rl_dopf_admm/src/adaptive/adaptive_base_functions.jl")

case_path = "data/case118_3.m"
data = parse_file(case_path)
tau_inc_values = [0, 0.0005, 0.001, 0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 0.7, 1]  
tau_dec_values = [0, 0.0005, 0.001, 0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 0.7, 1] 
tau_inc_values = 0.0001:0.005:0.08 #[0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
tau_dec_values = 0.0001:0.005:0.08 #[0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]

rng = StableRNG(123)
    
env = AdaptiveADMMEnv(data, tau_inc_values, tau_dec_values, rng, baseline_alpha_pq = 400, baseline_alpha_vt = 4000, tau_update_freq = 10)
ns, na = length(state(env)), length(action_space(env))

run_num = 2
sub_num = 18000
agent = BSON.load("data/saved_agents/adaptive_agent_$run_num"*"_$sub_num.bson")["agent"]

Qt = agent.policy.learner.target_approximator
polt_data_area, statet_trace = test_policy(Qt)

Q = agent.policy.learner.approximator 
pol_data_area, state_trace = test_policy(Q)
policyt_iterations = polt_data_area[1]["counter"]["iteration"]
policy_iterations = pol_data_area[1]["counter"]["iteration"]
println("Baseline: ", 143, "  policy with target: ", policyt_iterations, "  policy not target: ", policy_iterations)

ep_rewards = [r for r in agent.trajectory.traces.reward if r > 10]
plot(ep_rewards)