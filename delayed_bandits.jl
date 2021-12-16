
Base.@kwdef mutable struct DelayedBanditsEnv <: AbstractEnv

    reward_distributions::Vector{<:Distribution}
    delay_distribution::Distribution
    n_steps::Int64
    n_episodes::Int64
    rng::AbstractRNG
    # cache
    reward::Float64
    delayed_rewards::Vector{Float64}
    curent_step::Int64
    is_terminated::Bool
end

function DelayedBanditsEnv(; 
                            k = 10, 
                            reward_distributions, 
                            delay_distribution, 
                            n_steps,
                            n_episodes, 
                            rng = Random.GLOBAL_RNG
                            ) 
    return DelayedBanditsEnv(
                            reward_distributions, 
                            delay_distribution, 
                            n_steps,
                            n_episodes, 
                            rng, 
                            0.0,
                            zeros(n_steps),
                            0, 
                            false
                            )
end

RLBase.action_space(env::DelayedBanditsEnv) = Base.OneTo(length(env.reward_distributions))

function (env::DelayedBanditsEnv)(action)

	env.curent_step += 1

    t_delay = rand(env.rng, env.delay_distribution)

    if env.curent_step + t_delay <= env.n_steps
        env.delayed_rewards[env.curent_step + t_delay] += rand(env.rng, 
                                                            env.reward_distributions[action])
    end

    env.reward = env.delayed_rewards[env.curent_step]

    env.is_terminated = env.curent_step == env.n_steps ? true : false
end

RLBase.is_terminated(env::DelayedBanditsEnv) = env.is_terminated

RLBase.reward(env::DelayedBanditsEnv) = env.reward

RLBase.state(env::DelayedBanditsEnv) = 1

RLBase.state_space(env::DelayedBanditsEnv) = Base.OneTo(1)

function RLBase.reset!(env::DelayedBanditsEnv)
	env.curent_step = 0
    env.delayed_rewards = zeros(env.n_steps)
    env.is_terminated = false
end

Random.seed!(env::DelayedBanditsEnv, x) = seed!(env.rng, x)

RLBase.NumAgentStyle(::DelayedBanditsEnv) = SINGLE_AGENT
RLBase.DynamicStyle(::DelayedBanditsEnv) = SEQUENTIAL
RLBase.ActionStyle(::DelayedBanditsEnv) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::DelayedBanditsEnv) = IMPERFECT_INFORMATION  # the distribution of noise and original reward is unknown to the agent
RLBase.StateStyle(::DelayedBanditsEnv) = Observation{Int}()
RLBase.RewardStyle(::DelayedBanditsEnv) = STEP_REWARD
RLBase.UtilityStyle(::DelayedBanditsEnv) = GENERAL_SUM
RLBase.ChanceStyle(::DelayedBanditsEnv) = STOCHASTIC  # the same action lead to different reward each time.
