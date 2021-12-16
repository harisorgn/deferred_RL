
Base.@kwdef mutable struct DeferredBanditsEnv <: AbstractEnv

    reward_distributions::Vector{<:Distribution}
    n_steps::Int64
    n_episodes::Int64
    rng::AbstractRNG
    # cache
    reward::Float64
    deferred_reward::Float64
    curent_step::Int64
    is_terminated::Bool
end

function DeferredBanditsEnv(; 
                            k = 10, 
                            reward_distributions, 
                            n_steps, 
                            n_episodes, 
                            rng = Random.GLOBAL_RNG
                            )
    return DeferredBanditsEnv(
                            reward_distributions, 
                            n_steps, 
                            n_episodes,
                            rng, 
                            0.0, 
                            0.0, 
                            0, 
                            false
                            )
end

function deferred_reward_hook(t, agent, env)

    actions_v = agent.trajectory[:action][end-t+1:end]

    env.deferred_reward = map(a -> count(x -> x == a, actions_v) *a*2.0, RLBase.action_space(env)) |>
                            sum |>
                            x -> x / length(actions_v)
end

RLBase.action_space(env::DeferredBanditsEnv) = Base.OneTo(length(env.reward_distributions))

function (env::DeferredBanditsEnv)(action)

	env.curent_step += 1

    env.reward = rand(env.rng, env.reward_distributions[action]) + env.deferred_reward

    env.deferred_reward = 0.0
    env.is_terminated = env.curent_step == env.n_steps ? true : false
end

RLBase.is_terminated(env::DeferredBanditsEnv) = env.is_terminated

RLBase.reward(env::DeferredBanditsEnv) = env.reward

RLBase.state(env::DeferredBanditsEnv) = 1

RLBase.state_space(env::DeferredBanditsEnv) = Base.OneTo(1)

function RLBase.reset!(env::DeferredBanditsEnv)
	env.curent_step = 0
    env.is_terminated = false
end

Random.seed!(env::DeferredBanditsEnv, x) = seed!(env.rng, x)

RLBase.NumAgentStyle(::DeferredBanditsEnv) = SINGLE_AGENT
RLBase.DynamicStyle(::DeferredBanditsEnv) = SEQUENTIAL
RLBase.ActionStyle(::DeferredBanditsEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::DeferredBanditsEnv) = IMPERFECT_INFORMATION  # the distribution of noise and original reward is unknown to the agent
RLBase.StateStyle(::DeferredBanditsEnv) = Observation{Int}()
RLBase.RewardStyle(::DeferredBanditsEnv) = STEP_REWARD
RLBase.UtilityStyle(::DeferredBanditsEnv) = GENERAL_SUM
RLBase.ChanceStyle(::DeferredBanditsEnv) = STOCHASTIC  # the same action lead to different reward each time.
