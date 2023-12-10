using Pkg 
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
include("ac_admm_env.jl")

rng = StableRNG(123)

case_path = "data/case118_3.m"
data = parse_file(case_path)

env = ACADMMEnv(data, rng, baseline_alpha_pq = 400, baseline_alpha_vt = 4000, alpha_update_freq = 20, pq_bounds=[100.0,2000.0], vt_bounds=[2000.0,6000.0])
ns, na = length(state(env)), length(action_space(env))

run_num = 4

agent = Agent(
    policy = PPOPolicy(
        approximator = ActorCritic(
            actor = Chain(
                Dense(ns, 256, relu; init = glorot_uniform(rng)),
                Dense(256, na; init = glorot_uniform(rng)),
            ),
            critic = Chain(
                Dense(ns, 256, relu; init = glorot_uniform(rng)),
                Dense(256, 1; init = glorot_uniform(rng)),
            ),
            optimizer = ADAM(1e-3),
        ),
        γ = 0.99f0,
        λ = 0.95f0,
        clip_range = 0.1f0,
        max_grad_norm = 0.5f0,
        n_epochs = 4,
        n_microbatches = 4,
        actor_loss_weight = 1.0f0,
        critic_loss_weight = 0.5f0,
        entropy_loss_weight = 0.001f0,
        update_freq = 2,
    ),
    trajectory = PPOTrajectory(;
    capacity = 10000,
    state = Matrix{Float32} => (ns, 1),
    action = Vector{Int} => (na,1),
    action_log_prob = Vector{Float32} => (1,),
    reward = Vector{Float32} => (1,),
    terminal = Vector{Bool} => (1,),
),
)

hook = ComposedHook(TotalRewardPerEpisode())
run(agent, env, StopAfterStep(20), hook)

# using Plots
# plot(hook[1].rewards, xlabel="Episode", ylabel="Reward", label="")
# savefig("data/figs/reward_vs_episode_$run_num.png")
# # dictionary to write
# reward_dict = Dict("rewards" => hook[1].rewards)
# # pass data as a json string (how it shall be displayed in a file)
# stringdata = JSON.json(reward_dict)
# # write the file with the stringdata variable information
# open("data/rewards/trial_$run_num.json", "w") do f
#         write(f, stringdata)
#      end

# #Compare to baseline 
# base_data_area = test_baseline()
# baseline_iterations = base_data_area[1]["counter"]["iteration"]
# Qt = agent.policy.learner.target_approximator
# polt_data_area, statet_trace, alphat_trace = test_policy(Qt)

# Q = agent.policy.learner.approximator 
# pol_data_area, state_trace, alpha_trace = test_policy(Q)
# policyt_iterations = polt_data_area[1]["counter"]["iteration"]
# policy_iterations = pol_data_area[1]["counter"]["iteration"]
# println("Baseline: ", baseline_iterations, "  policy with target: ", policyt_iterations, "  policy not target: ", policy_iterations)

# bson("data/saved_agents/agent_$run_num.bson", Dict("agent" => agent))
# bson("data/trained_Qs/trial_$run_num.bson", Dict("Q" => Q, "Qt" => Qt))