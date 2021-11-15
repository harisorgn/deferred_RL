using Random
using ReinforcementLearning
using Flux
using UnicodePlots
using Distributions

include("do_every_step_episode_hook.jl")
include("deferred_bandits.jl")
include("experience_priority_sampling_model.jl")
include("tabular_bias_approximator.jl")
include("Delta_approximator.jl")
include("Delta_learner.jl")
include("Delta_agent.jl")

n_bandits = 10
n_episodes = 1
n_steps_per_episode = 50

reward_distributions = fill(DiscreteNonParametric([0.0],[1]), n_bandits)

env = DeferredBanditsEnv(; k=n_bandits,
                        reward_distributions=reward_distributions, 
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
p = stairs(1:length(q[:]), q[:], style=:post, ylabel="Q")
for x in RLBase.action_space(env)
    annotate!(p, x+0.5, 0, string(x))
end
println(p)

b = agent.policy.learner.offline_approximator.table[:]
push!(b, 0.0)
p = stairs(1:length(b[:]), b[:], style=:post, ylabel="bias")
for x in RLBase.action_space(env)
    annotate!(p, x+0.5, 0, string(x))
end
println(p)
