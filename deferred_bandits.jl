
Base.@kwdef mutable struct DeferredBanditsEnv <: AbstractEnv
    true_rewards::Vector{Float64}
    true_values::Vector{Float64}
    n_steps::Int64
    rng::AbstractRNG
    # cache
    reward::Float64
    deferred_reward::Float64
    curent_step::Int64
    is_terminated::Bool
end

function DeferredBanditsEnv(; k = 10, true_rewards = zeros(k), n_steps, rng = Random.GLOBAL_RNG)
    true_values = true_rewards .+ randn(rng, k)
    DeferredBanditsEnv(true_rewards, true_values, n_steps, rng, 0.0, 0.0, 0, false)
end

function deferred_reward_hook(t, agent, env)
    env.deferred_reward = count(x -> x == 1, agent.trajectory[:action]) * 1.0
end

RLBase.action_space(env::DeferredBanditsEnv) = Base.OneTo(length(env.true_values))

function (env::DeferredBanditsEnv)(action)

	env.curent_step += 1

    env.reward = env.true_values[action] + randn(env.rng) + env.deferred_reward

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
    env.true_values = env.true_rewards .+ randn(env.rng, length(env.true_rewards))
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
