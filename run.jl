using MacroTools: @forward

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
