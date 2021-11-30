using Random
using ReinforcementLearning
using Flux
using Distributions

include("do_every_step_episode_hook.jl")
include("deferred_bandits.jl")
include("experience_priority_sampling_model.jl")
include("tabular_bias_approximator.jl")
include("Delta_approximator.jl")
include("Delta_learner.jl")
include("Delta_agent.jl")
include("plot.jl")

n_bandits = 10
n_episodes = 400
n_steps_per_episode = 20
n_runs = 1000

reward_distributions = fill(DiscreteNonParametric([0.0],[1.0]), n_bandits)
#reward_distributions = fill(DiscreteNonParametric([0.0, 1.0],[0.5, 0.5]), n_bandits)
#reward_distributions = fill(Normal(0,1), n_bandits)

N_offline_samples = 10

η_Q_Delta = 0.2
η_Q_mc = 0.2

η_Δ = 0.5
η_b = 1.0/N_offline_samples

r_Delta = zeros(n_runs, n_episodes)
r_no_Delta = zeros(n_runs, n_episodes)
r_mc = zeros(n_runs, n_episodes)

Q = [Vector{Float64}() for _=1:n_runs]
b = [Vector{Float64}() for _=1:n_runs]
Q_mc = [Vector{Float64}() for _=1:n_runs]
Δ = zeros(n_runs, n_episodes)

explorer_soft = WeightedSoftmaxExplorer()
explorer_ε = EpsilonGreedyExplorer(0.1)

h = DoEveryNStepEveryEpisode(deferred_reward_hook; n=n_steps_per_episode-1)

for i = 1:n_runs

    env = DeferredBanditsEnv(; 
                            k=n_bandits,
                            reward_distributions=reward_distributions, 
                            n_steps=n_steps_per_episode
                            )

    learner = DeltaLearner(; 
                            n_state=length(state_space(env)),
                            n_action=length(action_space(env)),
                            η_Q=η_Q_Delta,
                            η_Δ=η_Δ,
                            η_b=η_b,
                            γ=0.0
                            )
    
    learner_no_Delta = DeltaLearner(; 
                                    n_state=length(state_space(env)),
                                    n_action=length(action_space(env)),
                                    η_Q=η_Q_Delta,
                                    η_Δ=0.0,
                                    η_b=0.0,
                                    γ=0.0,
                                    )

    learner_mc = MonteCarloLearner(; 
                                approximator=TabularQApproximator(; 
                                                                n_state=length(state_space(env)),
                                                                n_action=length(action_space(env)),
                                                                opt=Descent(η_Q_mc)
                                                                ),
                                γ=1.0,
                                kind=EVERY_VISIT
                                )

    agent = DeltaAgent(
                        QBasedPolicy(learner, explorer_soft), 
                        ExperiencePrioritySamplingModel(; N_samples=N_offline_samples),
                        VectorSARTTrajectory()
                        )

    agent_no_Delta = DeltaAgent(
                                QBasedPolicy(learner_no_Delta, explorer_ε), 
                                ExperiencePrioritySamplingModel(; N_samples=0),
                                VectorSARTTrajectory()
                                )

    mc_agent = Agent(
                    QBasedPolicy(learner_mc, explorer_ε), 
                    VectorSARTTrajectory()
                    )

    rand_agent = Agent(
                    RandomPolicy(RLBase.action_space(env)),
                    VectorSARTTrajectory()
                    )

    for j = 1:n_episodes

        run(
            agent,
            env,
            StopAfterEpisode(1; is_show_progress=false),
            h
            )

        run(
            agent_no_Delta,
            env,
            StopAfterEpisode(1; is_show_progress=false),
            h
            )

        run(
            mc_agent,
            env,
            StopAfterEpisode(1; is_show_progress=false),
            h
            )

        run(
            rand_agent,
            env,
            StopAfterEpisode(1; is_show_progress=false),
            h
            )

        r_rand = sum(rand_agent.trajectory[:reward][end-n_steps_per_episode+1:end])

        r_Delta[i,j] = sum(agent.trajectory[:reward]) - r_rand
        r_no_Delta[i,j] = sum(agent_no_Delta.trajectory[:reward]) - r_rand
        r_mc[i,j] = sum(mc_agent.trajectory[:reward]) - r_rand

        Δ[i,j] = agent.policy.learner.Δ_approximator.Δ
    end

    Q[i] = agent.policy.learner.approximator.table[:,1]
    b[i] = agent.policy.learner.offline_approximator.table[:,1]
    Q_mc[i] = mc_agent.policy.learner.approximator.table[:,1]
end

r = sum(r_Delta, dims=2)[:]
p = sortperm(r, rev=true)
r = r[p]
Q = Q[p]
b = b[p]
Δ = Δ[p,:]

r = sum(r_mc, dims=2)[:]
p = sortperm(r, rev=true)
Q_mc = Q_mc[p]

plot_band(r_Delta, r_mc, r_no_Delta)


