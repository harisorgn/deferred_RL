
Base.@kwdef struct DeltaAgent{M,T} <: AbstractPolicy
    policy::QBasedPolicy{DeltaLearner}
    model::M
    trajectory::T
    replay_steps::Int = 10
end

(p::DeltaAgent)(env::AbstractEnv) = p.policy(env)

function (agent::DeltaAgent)(stage::AbstractStage, env::AbstractEnv)
    update!(agent.trajectory, agent.policy, env, stage)
    Delta_update!(agent, env, stage)
end

function (agent::DeltaAgent)(stage::PreActStage, env::AbstractEnv, action)
    update!(agent.trajectory, agent.policy, env, stage, action)
    Delta_update!(agent, env, stage)
end

function Delta_update!(agent, env, stage)
    # 1. model learning
    update!(agent.model, agent.trajectory, agent.policy, env, stage)
    # 2. direct learning
    update!(agent.policy, agent.trajectory, env, stage)
    # 3. policy learning
    for _ in 1:agent.replay_steps
        update!(agent.policy, agent.model, agent.trajectory, env, stage)
    end
end

# 1. model learning
# By default we do nothing
function RLBase.update!(
    ::AbstractEnvironmentModel,
    ::AbstractTrajectory,
    ::AbstractPolicy,
    ::AbstractEnv,
    ::AbstractStage,
) end

# 3. policy learning
function RLBase.update!(
    ::AbstractPolicy,
    ::AbstractEnvironmentModel,
    ::AbstractTrajectory,
    ::AbstractEnv,
    ::AbstractStage,
) end