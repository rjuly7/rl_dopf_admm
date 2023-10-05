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
include("adaptive/adaptive_admm_env.jl")

rng = StableRNG(123)

case_path = "data/case14.m"
data = parse_file(case_path)
param_values = [
    [0.1, 0.2], #eta_inc 
    [0.1, 0.2], #eta_dec 
    [1.5, 2.0],  #mu_inc 
    [1.5, 2.0]   #mu_dec 
]

env = ADMMEnv(data, param_values, rng, baselines = [0.2,0.2,2.0,2.0], alpha_update_freq = 10)
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
            batch_size = 200,
            update_horizon = 1,
            min_replay_history = 100,
            update_freq = 1,
            target_update_freq = 100,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.05,
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
run(agent, env, StopAfterStep(100), hook)

using Plots
plot(hook[1].rewards, xlabel="Episode", ylabel="Reward", label="")
# dictionary to write
reward_dict = Dict("rewards" => hook[1].rewards)
# pass data as a json string (how it shall be displayed in a file)
run_num = 1
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

