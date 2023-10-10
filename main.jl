using Random
using ReinforcementLearning
using Flux
using Distributions
using LaTeXStrings
using Serialization

import StatsBase: sample
using StatsBase: Weights
using Statistics

using MacroTools: @forward

using CairoMakie
using ColorSchemes

include("do_every_step_episode_hook.jl")
include("deferred_bandits.jl")
include("delayed_bandits.jl")
include("experience_priority_sampling_model.jl")
include("tabular_bias_approximator.jl")
include("Delta_approximator.jl")
include("Delta_learner.jl")
include("Delta_agent.jl")
include("run.jl")

n_bandits = 10
n_steps_per_episode = 50
n_episodes = 200
n_runs = 1000

learning_rates = vcat([0.01, 0.02], 0.05:0.1:1.0)

h = DoEveryNStepEveryEpisode(deferred_reward_hook; n=n_steps_per_episode - 1)

envs_DB = reduce(vcat, [
        get_DB1_envs(n_bandits, n_steps_per_episode, n_episodes),
        get_DB2_envs(n_bandits, n_steps_per_episode, n_episodes),
        get_DB3_envs(n_bandits, n_steps_per_episode, n_episodes)
    ]
)
dists_DB = map(env -> first(env.reward_distributions), envs_DB)
R_DB = run_environments(envs_DB, h, learning_rates; n_episodes, n_runs)

envs_TB = get_TB_envs(n_bandits, n_steps_per_episode, n_episodes)
dists_TB = map(env -> first(env.reward_distributions), envs_TB)
R_TB = run_environments(envs_TB, EmptyHook(), learning_rates; n_episodes, n_runs)

R = vcat(R_DB, R_TB)
dists = vcat(dists_DB, dists_TB)

serialize("R.jls", R)
serialize("dists.jls", dists)
