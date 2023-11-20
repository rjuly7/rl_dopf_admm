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

run_nums = [6,6,6,6,5,7,7]
num_iters = [5000, 8000, 12000, 16000, 20000, 24000, 30000]
approx_iters = []
target_iters = []
alpha_traces = []

for i in eachindex(run_nums)
    run_num = run_nums[i]
    sub_num = num_iters[i]
    
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
    push!(approx_iters, policy_iterations)
    push!(target_iters, policyt_iterations)
    push!(alpha_traces, Dict("approx" => alpha_trace, "target" => alphat_trace))
end

bson("data/rewards/iter_vs_time.bson", Dict("approx_iters" => approx_iters, "target_iters" => target_iters, "alpha_traces" => alpha_traces))

plot_colors = palette(:tab10)
baseline = []
plot_costs = []
markershape = []
markerstrokewidth = []
line_plot_costs = Dict("x" => [], "y" => [])
marker_plot_costs = Dict("x" => [], "y" => [])
for i in eachindex(run_nums)
    if run_nums[i] == 5 || run_nums[i] == 6
        push!(line_plot_costs["x"], num_iters[i])
        push!(line_plot_costs["y"], target_iters[i])
        push!(marker_plot_costs["x"], num_iters[i])
        push!(marker_plot_costs["y"], target_iters[i])
        push!(baseline,143)
    end
end

plot(line_plot_costs["x"],line_plot_costs["y"],label="RL policy",color=plot_colors[1],linewidth=2.4,alpha=0.7,xlabel="Number of steps", ylabel="Iterations to converge")
plot!(line_plot_costs["x"],baseline,label="Baseline",color=plot_colors[2],linewidth=2.4,alpha=0.7)
scatter!(marker_plot_costs["x"],marker_plot_costs["y"],markercolor=plot_colors[1],markershape=:circle,markerstrokewidth=0.8,label="")
savefig("data/figs/iter_vs_time.png")

plot_colors = palette(:tab10)
baseline = []
plot_costs = []
markershape = []
markerstrokewidth = []
line_plot_costs = Dict("x" => [], "y" => [])
marker_plot_costs = Dict("x" => [], "y" => [])
for i in eachindex(run_nums)
    if (run_nums[i] == 5 && num_iters[i] == 20000) || run_nums[i] == 7
        push!(line_plot_costs["x"], num_iters[i])
        push!(line_plot_costs["y"], target_iters[i])
        push!(marker_plot_costs["x"], num_iters[i])
        push!(marker_plot_costs["y"], target_iters[i])
        push!(baseline,143)
    end
end

plot(line_plot_costs["x"],line_plot_costs["y"],label="RL policy",color=plot_colors[1],linewidth=2.4,alpha=0.7,xlabel="Number of steps", ylabel="Iterations to converge")
plot!(line_plot_costs["x"],baseline,label="Baseline",color=plot_colors[2],linewidth=2.4,alpha=0.7)
scatter!(marker_plot_costs["x"],marker_plot_costs["y"],markercolor=plot_colors[1],markershape=:circle,markerstrokewidth=0.8,label="")
savefig("data/figs/iter_vs_final_steps.png")
