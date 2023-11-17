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
include("original/admm_env.jl")

rng = StableRNG(123)

case_path = "data/case118_3.m"
data = parse_file(case_path)
pq_alpha_values = [200, 350, 400, 450, 600, 800]
vt_alpha_values = [2800, 3200, 3600, 3800, 4000, 4200, 4600]

pq_alpha_values = 200:100:800
vt_alpha_values = 2800:200:4800

env = ADMMEnv(data, pq_alpha_values, vt_alpha_values, rng, baseline_alpha_pq = 400, baseline_alpha_vt = 4000, alpha_update_freq = 5)
ns, na = length(state(env)), length(action_space(env))

run_num = 6
sub_num = 12000

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
            batch_size = 100,
            update_horizon = 1,
            min_replay_history = 100,
            update_freq = 2,
            target_update_freq = 50,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_init = 1,
            ϵ_stable = 0.05,
            decay_steps = 19000,
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

# dictionary to write
reward_dict = Dict("rewards" => hook[1].rewards)
# pass data as a json string (how it shall be displayed in a file)
stringdata = JSON.json(reward_dict)
# write the file with the stringdata variable information
open("/storage/scratch1/8/rharris94/rl_dopf_admm/data/rewards/trial_$run_num.json", "w") do f
        write(f, stringdata)
     end

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

bson("/storage/scratch1/8/rharris94/rl_dopf_admm/data/saved_agents/agent_$run_num"*"_$sub_num.bson", Dict("agent" => agent))
bson("/storage/scratch1/8/rharris94/rl_dopf_admm/data/trained_Qs/trial_$run_num"*"_$sub_num.bson", Dict("Q" => Q, "Qt" => Qt))