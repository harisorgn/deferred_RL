using Random
using ReinforcementLearning
using Flux
using Distributions
using LaTeXStrings
using Serialization

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
n_steps_per_episode = 50
n_episodes = 300
n_runs = 1000


dist_v = [
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
        for r_dist in dist_v
        ]
h = DoEveryNStepEveryEpisode(
                            deferred_reward_hook; 
                            n=n_steps_per_episode-1
                            )

=#


env_v = fill(
            DelayedBanditsEnv(; 
                            k=10, 
                            reward_distributions=[Normal(i,1) for i=1:n_bandits], 
                            delay_distribution=Poisson(4), 
                            n_steps=n_steps_per_episode, 
                            n_episodes=n_episodes,
                            rng=Random.GLOBAL_RNG
                            ),
            1
            )
h = EmptyHook()


η_v = 0.05:0.1:1.0
H = repeat(η_v, inner=length(env_v))
E = repeat(env_v, length(η_v))

R = map(zip(E,H)) do (env, η)

        # WeightedSoftmaxExplorer()
        # EpsilonGreedyExplorer(0.1)

        agent_Delta = DeltaAgent(
                        QBasedPolicy(
                                    DeltaLearner(; 
                                                n_state=length(state_space(env)),
                                                n_action=length(action_space(env)),
                                                η_Q=η,
                                                η_Δ=0.02,
                                                η_b=0.1,
                                                γ=0.0
                                                ), 
                                    EpsilonGreedyExplorer(0.1)
                                    ), 
                        ExperiencePrioritySamplingModel(; 
                                                    N_samples=10
                                                    ),
                        VectorSARTTrajectory()
                        )

        agent_no = DeltaAgent(
                        QBasedPolicy(
                                    DeltaLearner(; 
                                                n_state=length(state_space(env)),
                                                n_action=length(action_space(env)),
                                                η_Q=η,
                                                η_Δ=0.0,
                                                η_b=0.0,
                                                γ=0.0
                                                ), 
                                    EpsilonGreedyExplorer(0.1)
                                    ), 
                        ExperiencePrioritySamplingModel(; 
                                                    N_samples=0
                                                    ),
                        VectorSARTTrajectory()
                        )

        agent_MC = Agent(
                        QBasedPolicy(
                                MonteCarloLearner(; 
                                    approximator=TabularQApproximator(; 
                                                        n_state=length(state_space(env)),
                                                        n_action=length(action_space(env)),
                                                        opt=Descent(η)
                                                        ),
                                    γ=1.0,
                                    kind=EVERY_VISIT
                                ), 
                                EpsilonGreedyExplorer(0.1)
                                ), 
                        VectorSARTTrajectory()
                        )

        R_env = run_agent.(
                        [agent_Delta, agent_no, agent_MC], 
                        Ref(env); 
                        n_runs=n_runs, 
                        h=h
                        )

        reduce((x,y)->cat(x,y; dims=3), R_env)
    end

R = reduce((x,y)->cat(x,y; dims=4), R)

serialize("R_delayed_new.jls", R)
serialize("H_delayed_new.jls", H)

