
Base.@kwdef mutable struct DeltaPolicy{Q<:AbstractLearner, 
										E<:AbstractExplorer} <: AbstractPolicy
    online_learner::Q
    offline_learner::Q
    explorer::E
end

(π::DeltaPolicy)(env) = π(env, ActionStyle(env), action_space(env))

(π::DeltaPolicy)(env, ::MinimalActionSet, ::Base.OneTo) = π.explorer(π.online_learner(env) + 
																	π.offline_learner(env))
(π::DeltaPolicy)(env, ::FullActionSet, ::Base.OneTo) =
    π.explorer(π.online_learner(env) + π.offline_learner(env)), legal_action_space_mask(env))

(π::DeltaPolicy)(env, ::MinimalActionSet, A) = A[π.explorer(π.online_learner(env) + π.offline_learner(env))]

(π::DeltaPolicy)(env, ::FullActionSet, A) =
    A[π.explorer(π.online_learner(env) + π.offline_learner(env), legal_action_space_mask(env))]

RLBase.prob(p::DeltaPolicy, env::AbstractEnv) = prob(p, env, ActionStyle(env))

RLBase.prob(p::DeltaPolicy, env::AbstractEnv, ::MinimalActionSet) =
    prob(p.explorer, p.online_learner(env) + p.offline_learner(env))

RLBase.prob(p::DeltaPolicy, env::AbstractEnv, ::FullActionSet) =
    prob(p.explorer, p.online_learner(env) + p.offline_learner(env), legal_action_space_mask(env))

function RLBase.prob(p::DeltaPolicy, env::AbstractEnv, action)
    A = action_space(env)
    P = prob(p, env)
    if P isa Distribution
        P = probs(P)
    end
    @assert length(A) == length(P)
    if A isa Base.OneTo
        P[action]
    else
        for (a, p) in zip(A, P)
            if a == action
                return p
            end
        end
        @error "action[$action] is not found in action space[$(action_space(env))]"
    end
end

@forward DeltaPolicy.learner RLBase.priority

function RLBase.update!(
    p::DeltaPolicy,
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    update!(p.online_learner, t, e, s)
    update!(p.offline_learner, t, e, s)
end

function check(p::DeltaPolicy, env::AbstractEnv)
    A = action_space(env)
    if (A isa AbstractVector && A == 1:length(A)) ||
       (A isa Tuple && A == Tuple(1:length(A)))
        # this is expected
    else
        @warn "Applying a DeltaPolicy to an environment with a unknown action space. Maybe convert the environment with `discrete2standard_discrete` in ReinforcementLearningEnvironments.jl first or redesign the environment."
    end

    check(p.online_learner, env)
    check(p.explorer, env)
end