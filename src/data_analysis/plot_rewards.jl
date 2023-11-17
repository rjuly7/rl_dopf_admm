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

case_path = "data/case118_3.m"
data = parse_file(case_path)

rng = StableRNG(123)

run_num = 7
sub_num = 30000

if run_num == 5 || run_num == 6
    pq_alpha_values = 200:100:800
    vt_alpha_values = 2800:200:4800
elseif run_num == 7
    pq_alpha_values = 200:50:800
    vt_alpha_values = 2800:100:4800
end

env = ADMMEnv(data, pq_alpha_values, vt_alpha_values, rng, baseline_alpha_pq = 400, baseline_alpha_vt = 4000, alpha_update_freq = 5)
ns, na = length(state(env)), length(action_space(env))

agent = BSON.load("data/saved_agents/agent_$run_num"*"_$sub_num.bson")["agent"]

#Compare to baseline 
base_data_area = test_baseline()
baseline_iterations = base_data_area[1]["counter"]["iteration"]
Qt = agent.policy.learner.target_approximator
polt_data_area, statet_trace = test_policy(Qt)

Q = agent.policy.learner.approximator 
pol_data_area, state_trace = test_policy(Q)
policyt_iterations = polt_data_area[1]["counter"]["iteration"]
policy_iterations = pol_data_area[1]["counter"]["iteration"]
println("Baseline: ", baseline_iterations, "  policy with target: ", policyt_iterations, "  policy not target: ", policy_iterations)

bson("data/saved_agents/agent_$run_num.bson", Dict("agent" => agent))
bson("data/trained_Qs/trial_$run_num.bson", Dict("Q" => Q, "Qt" => Qt))