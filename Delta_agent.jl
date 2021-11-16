
Base.@kwdef struct DeltaAgent{T} <: AbstractPolicy
    policy::QBasedPolicy{<:DeltaLearner}
    model::ExperiencePrioritySamplingModel
    trajectory::T
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

function (agent::DeltaAgent)(::PreEpisodeStage, ::AbstractEnv)
    empty!(agent.model.experiences)
end

function Delta_update!(agent, env, stage)
    # experience accumulation
    update!(agent.model, agent.trajectory, agent.policy, env, stage)
    # online learning
    update!(agent.policy, agent.trajectory, env, stage)
    # offline learning
    update!(agent.policy, agent.model, agent.trajectory, env, stage)
end
