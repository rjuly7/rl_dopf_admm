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

case_path = "data/case118_3.m"
data = parse_file(case_path)
tau_inc_values = 0.0001:0.005:0.08 #[0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
tau_dec_values = 0.0001:0.005:0.08 #[0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]

env = AdaptiveADMMEnv(data, tau_inc_values, tau_dec_values, rng, baseline_alpha_pq = 400, baseline_alpha_vt = 4000, tau_update_freq = 10)
ns, na = length(state(env)), length(action_space(env))

run_num = 2
sub_num = 18000

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
                optimizer = ADAM(),
            ),
            target_approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, na; init = glorot_uniform(rng)),
                ),
                optimizer = ADAM(),
            ),
            loss_func = mse,
            stack_size = nothing,
            batch_size = 200,
            update_horizon = 1,
            min_replay_history = 450,
            update_freq = 2,
            target_update_freq = 50,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_init = 1,
            ϵ_stable = 0.02,
            decay_steps = 17000,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 50000,
        state = Vector{Float32} => (ns,),
    ),
)


agent.policy.learner.sampler.γ = 0.97 #vary between (0.8,0.99)
hook = ComposedHook(TotalRewardPerEpisode())
run(agent, env, StopAfterStep(sub_num), hook)

using Plots
plot(hook[1].rewards, xlabel="Episode", ylabel="Reward", label="")
savefig("data/figs/adaptive_reward_vs_episode_$run_num.png")
# dictionary to write
reward_dict = Dict("rewards" => hook[1].rewards)
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(reward_dict)
# write the file with the stringdata variable information
open("data/rewards/adaptive_trial_$run_num.json", "w") do f
        write(f, stringdata)
     end

#Compare to baseline 
base_data_area = test_baseline()
baseline_iterations = base_data_area[1]["counter"]["iteration"]
Qt = agent.policy.learner.target_approximator
polt_data_area, statet_trace = test_policy(Qt)

Q = agent.policy.learner.approximator 
pol_data_area, state_trace = test_policy(Q)
policyt_iterations = pol_data_area[1]["counter"]["iteration"]
policy_iterations = polt_data_area[1]["counter"]["iteration"]
println("Baseline: ", baseline_iterations, "  policy with target: ", policyt_iterations, "  policy not target: ", policy_iterations)

bson("data/saved_agents/adaptive_agent_$run_num"*"_$sub_num.bson", Dict("agent" => agent))
bson("data/trained_Qs/adaptive_trial_$run_num"*"_$sub_num.bson", Dict("Q" => Q, "Qt" => Qt))
