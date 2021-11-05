using Random
using ReinforcementLearning
using Flux
using UnicodePlots

include("do_every_step_episode_hook.jl")
include("deferred_bandits.jl")
include("experience_priority_sampling_model.jl")
include("tabular_bias_approximator.jl")
include("Delta_approximator.jl")
include("Delta_learner.jl")
include("Delta_agent.jl")

n_bandits = 10
n_episodes = 10
n_steps_per_episode = 50
#true_rewards = [5,5,5,0,0,2,2,2,0,0]
true_rewards = fill(5.0, n_bandits)

env = DeferredBanditsEnv(; k=n_bandits,
                        true_rewards=true_rewards, 
                        n_steps=n_steps_per_episode)

η_Q = 0.5
η_Δ = 0.5
η_b = 0.5
γ = 0.0

learner = DeltaLearner(; n_state=length(state_space(env)),
                        n_action=length(action_space(env)),
                        η_Q=η_Q,
                        η_Δ=η_Δ,
                        η_b=η_b,
                        γ=γ)
                        


explorer = WeightedSoftmaxExplorer()

#agent = Agent(QBasedPolicy(learner, explorer), 
#                VectorSARTTrajectory())

agent = DeltaAgent(QBasedPolicy(learner, explorer), 
                    ExperiencePrioritySamplingModel(; N_samples=10),
                    VectorSARTTrajectory())

h = DoEveryNStepEveryEpisode(deferred_reward_hook; n=n_steps_per_episode-1)

run(agent,
    env,
    StopAfterEpisode(n_episodes; is_show_progress = false),
    h)

q = agent.policy.learner.approximator.table[:]
push!(q, 0.0)
println(stairs(1:length(q[:]), q[:], style=:post, ylabel="Q"))

b = agent.policy.learner.offline_approximator.table[:]
push!(b, 0.0)
println(stairs(1:length(b[:]), b[:], style=:post, ylabel="bias"))

