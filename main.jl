using Random
using ReinforcementLearning
using Flux
using Distributions
using CairoMakie: lines!, band!, save, Figure, Axis

include("do_every_step_episode_hook.jl")
include("deferred_bandits.jl")
include("experience_priority_sampling_model.jl")
include("tabular_bias_approximator.jl")
include("Delta_approximator.jl")
include("Delta_learner.jl")
include("Delta_agent.jl")

n_bandits = 10
n_episodes = 500
n_steps_per_episode = 20

#reward_distributions = fill(DiscreteNonParametric([0.0],[1.0]), n_bandits)    
reward_distributions = fill(DiscreteNonParametric([0.0, 0.5],[0.5, 0.5]), n_bandits)
#reward_distributions = fill(Normal(0,1), n_bandits)

N_offline_samples = 10

η_Q_Delta = 0.2
η_Q_mc = 0.2

η_Δ = 0.5
η_b = 1.0/N_offline_samples
γ = 1.0

h = DoEveryNStepEveryEpisode(deferred_reward_hook; n=n_steps_per_episode-1)

n_runs = 1000
r_Delta = zeros(n_runs, n_episodes)
r_mc = zeros(n_runs, n_episodes)
r_rand = zeros(n_runs, n_episodes)

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
                            γ=γ
                            )
                            

    explorer_soft = WeightedSoftmaxExplorer()
    explorer_ε = EpsilonGreedyExplorer(0.1)

    agent = DeltaAgent(
                        QBasedPolicy(learner, explorer_ε), 
                        ExperiencePrioritySamplingModel(; N_samples=N_offline_samples),
                        VectorSARTTrajectory()
                        )

    mc_learner = MonteCarloLearner(; 
                                approximator=TabularQApproximator(; 
                                                                n_state=length(state_space(env)),
                                                                n_action=length(action_space(env)),
                                                                opt=Descent(η_Q_mc)
                                                                ),
                                kind=EVERY_VISIT
                                )

    mc_agent = Agent(
                    QBasedPolicy(mc_learner, explorer_ε), 
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

        r_Delta[i,j] = sum(agent.trajectory[:reward])
        r_mc[i,j] = sum(mc_agent.trajectory[:reward])
        r_rand[i,j] = sum(rand_agent.trajectory[:reward][end-n_steps_per_episode+1:end])
    end
end


f = Figure()
Axis(f[1, 1])

t = 1:n_episodes

μ_Delta = vec(mean(r_Delta, dims=1))
σ_Delta = vec(std(r_Delta, dims=1))

μ_mc = vec(mean(r_mc, dims=1))
σ_mc = vec(std(r_mc, dims=1))

μ_rand = vec(mean(r_rand, dims=1))
σ_rand = vec(std(r_rand, dims=1))

lines!(t, μ_Delta, color=:blue)
lines!(t, μ_mc, color=:red)
lines!(t, μ_rand, color=:green)

band!(t, μ_Delta + σ_Delta, μ_Delta - σ_Delta, color=(:blue,0.5))
band!(t, μ_mc + σ_mc, μ_mc - σ_mc, color=(:red,0.5))
band!(t, μ_rand + σ_rand, μ_rand - σ_rand, color=(:green,0.2))

save("res.png", f)

