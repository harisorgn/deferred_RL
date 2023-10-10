function reset!(agent::DeltaAgent)
	agent.policy.learner.approximator.table .= zero(
												eltype(agent.policy.learner.approximator.table)
												)
	agent.policy.learner.Δ_approximator.Δ = 0.0
	agent.policy.learner.offline_approximator.table .= zero(
														eltype(agent.policy.learner.offline_approximator.table)
														)
end

@forward Agent.policy reset!

function reset!(p::T) where T<:AbstractPolicy
	p.learner.approximator.table .= zero(eltype(p.learner.approximator.table))
end

function run_agent(agent, env; n_runs=1000, h=EmptyHook())

    R = zeros(n_runs, env.n_episodes)

    for i = 1:n_runs
        for j = 1:env.n_episodes

            rand_agent = Agent(
                            RandomPolicy(RLBase.action_space(env)),
                            VectorSARTTrajectory()
                            )
            
            run(
                agent,
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

            r_rand = sum(rand_agent.trajectory[:reward])

            R[i,j] = sum(agent.trajectory[:reward]) - r_rand
        end

        reset!(agent)
        RLBase.reset!(env)
    end
    return R
end

init_results_matrix(n_runs, n_episodes, n_param_values, n_agents) = zeros(n_runs, n_episodes, n_param_values, n_agents)

function get_DB1_envs(n_bandits=10, n_steps_per_episode=50, n_episodes=10)
    
    dist = fill(DiscreteNonParametric([0.0],[1.0]), n_bandits)

    env = DeferredBanditsEnv(; 
        k=n_bandits,
        reward_distributions=dist, 
        n_steps=n_steps_per_episode,
        n_episodes=n_episodes
    )

    return [env]
end

function get_DB2_envs(n_bandits=10, n_steps_per_episode=50, n_episodes=10)
    
    dists = [
        fill(DiscreteNonParametric([0.0, 1.0],[0.5, 0.5]), n_bandits),
        fill(DiscreteNonParametric([0.0, 5.0],[0.5, 0.5]), n_bandits),
        fill(DiscreteNonParametric([0.0, 1.0],[0.2, 0.8]), n_bandits),
    ]

    envs = [
        DeferredBanditsEnv(; 
            k=n_bandits,
            reward_distributions=dist, 
            n_steps=n_steps_per_episode,
            n_episodes=n_episodes
        )
        for dist in dists
    ]

    return envs
end

function get_DB3_envs(n_bandits=10, n_steps_per_episode=50, n_episodes=10)
    
    dists = [
        fill(Normal(0,1), n_bandits),
        fill(Normal(5,1), n_bandits),
        fill(Normal(0,3), n_bandits)
    ]

    envs = [
        DeferredBanditsEnv(; 
            k=n_bandits,
            reward_distributions=dist, 
            n_steps=n_steps_per_episode,
            n_episodes=n_episodes
        )
        for dist in dists
    ]

    return envs
end

function get_TB_envs(n_bandits=10, n_steps_per_episode=50, n_episodes=10)

    dists = [
        [Normal(i,0.3) for i=1:n_bandits],
        [Normal(i,1) for i=1:n_bandits],
        [Normal(2*i,0.3) for i=1:n_bandits]
    ]
    envs = [
            DelayedBanditsEnv(; 
                k=n_bandits, 
                reward_distributions=dist, 
                delay_distribution=Poisson(4), 
                n_steps=n_steps_per_episode, 
                n_episodes=n_episodes,
                rng=Random.GLOBAL_RNG
            )
        for dist in dists
    ]
            
    return envs
end

function run_environments(envs, hook, learning_rates; n_episodes=10, n_runs=10)
    
    Rs = [
        init_results_matrix(n_runs, n_episodes, length(learning_rates), 3)
        for _ in eachindex(envs)
    ]
    
    epsilon = 0.05

    for (k, env) in enumerate(envs)
        for (i, η) in enumerate(learning_rates)
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
                        EpsilonGreedyExplorer(epsilon)
                        ), 
                ExperiencePrioritySamplingModel(; N_samples=10),
                VectorSARTTrajectory()
            )

            agent_Q = DeltaAgent(
                QBasedPolicy(
                        DeltaLearner(; 
                                n_state=length(state_space(env)),
                                n_action=length(action_space(env)),
                                η_Q=η,
                                η_Δ=0.0,
                                η_b=0.0,
                                γ=0.0
                        ), 
                        EpsilonGreedyExplorer(epsilon)
                        ), 
                ExperiencePrioritySamplingModel(; N_samples=0),
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
                        EpsilonGreedyExplorer(epsilon)
                ), 
                VectorSARTTrajectory()
            )

            for (j, agent) in enumerate([agent_Delta, agent_MC, agent_Q])
                Rs[k][:,:,i,j] = run_agent(agent, env; n_runs=n_runs, h=hook)
            end
        end
    end

    return Rs
end