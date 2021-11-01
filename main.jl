using Random
using ReinforcementLearning
using Flux
using UnicodePlots

include("deferred_bandits.jl")
include("tabular_bias_approximator.jl")
include("Delta_approximator.jl")
include("Delta_learner.jl")

n_bandits = 2
n_episodes = 100
n_steps_per_episode = 150
#true_rewards = [5,5,5,0,0,2,2,2,0,0]
true_rewards = fill(5.0, n_bandits)

env = DeferredBanditsEnv(; k=n_bandits,
                        true_rewards=true_rewards, 
                        n_steps=n_steps_per_episode)

η_Q = 0.1
η_Δ = 0.1
η_b = 0.1
γ = 0.0

learner = DeltaLearner(; n_state=length(state_space(env)),
                        n_action=length(action_space(env)),
                        η_Q=η_Q,
                        η_Δ=η_Δ,
                        η_b=η_b,
                        γ=γ)
                        


explorer = WeightedSoftmaxExplorer()

agent = Agent(QBasedPolicy(learner, explorer), 
                VectorSARTTrajectory())

h = DoEveryNEpisode(deferred_reward_hook)
#h = DoEveryNStep(deferred_reward_hook; n=10)

run(agent,
    env,
    StopAfterEpisode(n_episodes; is_show_progress = false),
    h)

v = agent.policy.learner.bias_approximator.table[:]

println(agent.policy.learner.approximator.table[:])
println(agent.policy.learner.bias_approximator.table[:])
println(stairs(1:length(v[:]), v[:], style=:post))
