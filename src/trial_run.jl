path = "C:/Users/User/Documents/rl_admm"

cd(path)
using Pkg
Pkg.activate(".")

using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using StatsBase 
using PowerModelsADA
using Ipopt 
include("admm_env.jl")

rng = StableRNG(123)

case_path = "$path/data/case14.m"
data = parse_file(case_path)
alpha_values = [10, 50, 100, 250, 500, 1000, 1500, 2000]
#env = MountainCarEnv(; T = Float32, max_steps = 5000, rng = rng)
env = ADMMEnv(data, alpha_values)
ns, na = length(state(env)), length(action_space(env))

agent = Agent(
    policy = QBasedPolicy(
        learner = DQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, na; init = glorot_uniform(rng)),
                ),
                optimizer = Adam(),
            ),
            target_approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, 64, relu; init = glorot_uniform(rng)),
                    Dense(64, na; init = glorot_uniform(rng)),
                ),
                optimizer = Adam(),
            ),
            loss_func = mse,
            stack_size = nothing,
            batch_size = 32,
            update_horizon = 1,
            min_replay_history = 100,
            update_freq = 1,
            target_update_freq = 100,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 100,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 50000,
        state = Vector{Float32} => (ns,),
    ),
)
hook = ComposedHook(TotalRewardPerEpisode())
run(agent, env, StopAfterStep(500), hook)

@info "stats for BasicDQNLearner" avg_reward = mean(hook[1].rewards) avg_fps = 1 / mean(hook[2].times)
# ┌ Info: stats for BasicDQNLearner
# │   avg_reward = 107.43478260869566
# └   avg_fps = 531.283841452491

using Plots
plot(hook[1].rewards, xlabel="Episode", ylabel="Reward", label="")

