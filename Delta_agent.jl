
Base.@kwdef struct DeltaAgent{T} <: AbstractPolicy
    policy::QBasedPolicy{<:DeltaLearner}
    model::ExperiencePrioritySamplingModel
    trajectory::T
end

(p::DeltaAgent)(env::AbstractEnv) = p.policy(env)

function (agent::DeltaAgent)(stage::AbstractStage, env::AbstractEnv)
    RLBase.update!(agent.trajectory, agent.policy, env, stage)
    Delta_update!(agent, env, stage)
end

function (agent::DeltaAgent)(stage::PreActStage, env::AbstractEnv, action)
    RLBase.update!(agent.trajectory, agent.policy, env, stage, action)
    Delta_update!(agent, env, stage)
end

function (agent::DeltaAgent)(stage::PreEpisodeStage, env::AbstractEnv)
    RLBase.update!(agent.trajectory, agent.policy, env, stage)
    empty!(agent.model.experiences)
end

function Delta_update!(agent, env, stage)
    # experience accumulation
    RLBase.update!(agent.model, agent.trajectory, agent, env, stage)
    # online learning
    RLBase.update!(agent.policy, agent.trajectory, env, stage)
    # offline learning
    RLBase.update!(agent.policy, agent.model, agent.trajectory, env, stage)
end

function RLBase.priority(agent::DeltaAgent, transition::Tuple, env::AbstractEnv)

    L = agent.policy.learner

    if L.method == :SARS
        s, a, r, d, s′ = transition
        γ, Q = L.γ, L.approximator
        δ = d ? (r - Q(s, a)) : (r + γ^(L.n + 1) * maximum(Q(s′)) - Q(s, a))
        δ = [δ]  # must be broadcastable in Flux.Optimise
        Flux.Optimise.apply!(Q.optimizer, (s, a), δ)

        #return abs(δ[])
        #return 1.0
        return (1.0 - RLBase.prob(agent.policy, env, a))
    else
        @error "unsupported method"
    end
end