using Random
using ReinforcementLearning
using Flux
using Distributions

include("do_every_step_episode_hook.jl")
include("deferred_bandits.jl")
include("delayed_bandits.jl")
include("experience_priority_sampling_model.jl")
include("tabular_bias_approximator.jl")
include("Delta_approximator.jl")
include("Delta_learner.jl")
include("Delta_agent.jl")
include("run.jl")
include("plot.jl")

n_bandits = 10
n_steps_per_episode = 20
n_episodes = 300

d = [
    fill(DiscreteNonParametric([0.0],[1.0]), n_bandits),
    fill(DiscreteNonParametric([0.0, 1.0],[0.5, 0.5]), n_bandits),
    fill(Normal(0,1), n_bandits)
    ]
#=
env_v = [
        DeferredBanditsEnv(; 
                        k=n_bandits,
                        reward_distributions=r_dist, 
                        n_steps=n_steps_per_episode,
                        n_episodes=n_episodes
                        )
        for r_dist in d
        ]
=#

env_v = [
        DelayedBanditsEnv(; 
                        k=10, 
                        reward_distributions=[Normal(i,1) for i=1:n_bandits], 
                        delay_distribution=Poisson(2), 
                        n_steps=n_steps_per_episode, 
                        n_episodes=n_episodes,
                        rng = Random.GLOBAL_RNG
                        )
        ]

R = map(env_v) do env

        agent_v = [
                DeltaAgent(
                        QBasedPolicy(
                                    DeltaLearner(; 
                                                n_state=length(state_space(env)),
                                                n_action=length(action_space(env)),
                                                η_Q=0.1,
                                                η_Δ=0.2,
                                                η_b=0.2,
                                                γ=0.0
                                                ), 
                                    WeightedSoftmaxExplorer()
                                    ), 
                        ExperiencePrioritySamplingModel(; 
                                                    N_samples=10
                                                    ),
                        VectorSARTTrajectory()
                        ),
                DeltaAgent(
                        QBasedPolicy(
                                    DeltaLearner(; 
                                                n_state=length(state_space(env)),
                                                n_action=length(action_space(env)),
                                                η_Q=0.1,
                                                η_Δ=0.0,
                                                η_b=0.0,
                                                γ=0.0
                                                ), 
                                    WeightedSoftmaxExplorer()
                                    ), 
                        ExperiencePrioritySamplingModel(; 
                                                    N_samples=0
                                                    ),
                        VectorSARTTrajectory()
                        ),
                Agent(
                    QBasedPolicy(
                                MonteCarloLearner(; 
                                    approximator=TabularQApproximator(; 
                                                        n_state=length(state_space(env)),
                                                        n_action=length(action_space(env)),
                                                        opt=Descent(0.2)
                                                        ),
                                    γ=1.0,
                                    kind=EVERY_VISIT
                                ), 
                                EpsilonGreedyExplorer(0.1)
                                ), 
                    VectorSARTTrajectory()
                    )
                ]

        run_agents.(
                    agent_v, 
                    Ref(env); 
                    n_runs=1000, 
                    h=DoEveryNStepEveryEpisode(
                                            deferred_reward_hook; 
                                            n=n_steps_per_episode-1)
                                            )
    end

plot_agents(R, ["Delta", "no", "MC"], ["Delayed"])

