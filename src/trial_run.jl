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
include("admm_env.jl")

rng = StableRNG(123)

case_path = "data/case14.m"
data = parse_file(case_path)
pq_alpha_values = [200, 300, 400, 450, 500, 600, 700, 800, 900, 1000]
vt_alpha_values = [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5500, 6000, 7000]

env = ADMMEnv(data, pq_alpha_values, vt_alpha_values, rng, baseline_alpha_pq = 400, baseline_alpha_vt = 4000)
ns, na = length(state(env)), length(action_space(env))

agent = Agent(
    policy = QBasedPolicy(
        learner = DQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, na; init = glorot_uniform(rng)),
                ),
                optimizer = Adam(),
            ),
            target_approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, na; init = glorot_uniform(rng)),
                ),
                optimizer = Adam(),
            ),
            loss_func = mse,
            stack_size = nothing,
            batch_size = 75,
            update_horizon = 1,
            min_replay_history = 100,
            update_freq = 1,
            target_update_freq = 100,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.03,
            decay_steps = 1500,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 50000,
        state = Vector{Float32} => (ns,),
    ),
)
hook = ComposedHook(TotalRewardPerEpisode())
run(agent, env, StopAfterStep(2500), hook)

@info "stats for BasicDQNLearner" avg_reward = mean(hook[1].rewards) avg_fps = 1 / mean(hook[2].times)

using Plots
plot(hook[1].rewards, xlabel="Episode", ylabel="Reward", label="")
# dictionary to write
reward_dict = Dict("rewards" => hook[1].rewards)
# pass data as a json string (how it shall be displayed in a file)
run_num = 2
stringdata = JSON.json(reward_dict)
# write the file with the stringdata variable information
open("data/rewards/trial_$run_num.json", "w") do f
        write(f, stringdata)
     end

#Compare to baseline 
base_data_area = test_baseline()
baseline_iterations = base_data_area[1]["counter"]["iteration"]
Q = agent.policy.learner.target_approximator
pol_data_area = test_policy(Q)
policy_iterations = pol_data_area[1]["counter"]["iteration"]
println("Baseline: ", baseline_iterations, "  policy: ", policy_iterations)

bson("data/trained_Qs/trial_$run_num.bson", Dict("Q" => Q))