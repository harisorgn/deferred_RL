
using LinearAlgebra: dot
using Distributions: pdf

Base.@kwdef struct DeltaLearner <: AbstractLearner
    approximator::Union{TabularQApproximator,LinearQApproximator}
    Δ_approximator::DeltaApproximator
    offline_approximator::TabularBiasApproximator
    γ::Float64 = 1.0
    method::Symbol
    n::Int = 0
end

DeltaLearner(; n_state, n_action, Q_init=0.0, η_Q, Δ_init=0.0, η_Δ, b_init=0.0 , η_b, 
                γ=0.0, method=:SARS, n=0) =
    DeltaLearner(TabularQApproximator(; n_state=n_state,
                                        n_action=n_action,
                                        opt=Descent(η_Q)),
                DeltaApproximator(; η = η_Δ),
                TabularBiasApproximator(; n_state=n_state,
                                        n_action=n_action,
                                        η = η_b), 
                γ, 
                method, 
                n)

(L::DeltaLearner)(env::AbstractEnv) = L.approximator(state(env)) + L.offline_approximator(state(env))
(L::DeltaLearner)(s) = L.approximator(s) + L.offline_approximator(s)
(L::DeltaLearner)(s, a) = L.approximator(s, a) + L.offline_approximator(s, a)

## update policies

function RLBase.update!(
    p::QBasedPolicy{<:DeltaLearner},
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    if p.learner.method === :ExpectedSARSA && s === PRE_ACT_STAGE
        update!(p.learner, (t, pdf(prob(p, e))), e, s)
    else
        update!(p.learner, t, e, s)
    end
end


function RLBase.update!(
	L::DeltaLearner, 
	t::AbstractTrajectory, 
	::AbstractEnv, 
	s::PreActStage
)
    _update!(L, L.approximator, Val(L.method), t, s)
    #_update!(L, L.offline_approximator, Val(L.method), t, s)
end

function RLBase.update!(
    L::DeltaLearner,
    t::AbstractTrajectory,
    ::AbstractEnv,
    s::PostEpisodeStage,
)
    _update!(L, L.approximator, Val(L.method), t, s)
    #_update!(L, L.offline_approximator, Val(L.method), t, s)
end

# for ExpectedSARSA
function RLBase.update!(
    L::DeltaLearner,
    t::Tuple,
    ::AbstractEnv,
    s::Union{PreActStage,PostEpisodeStage},
)
    _update!(L, L.approximator, Val(L.method), t, s)
    #_update!(L, L.offline_approximator, Val(L.method), t, s)
end

# update trajectories
function RLBase.update!(
    t::AbstractTrajectory,
    ::Union{
        QBasedPolicy{<:DeltaLearner},
        NamedPolicy{<:QBasedPolicy{<:DeltaLearner}},
        VBasedPolicy{<:DeltaLearner},
    },
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end

# implementations

function _update!(
    L::DeltaLearner,
    ::Union{TabularQApproximator,LinearQApproximator},
    ::Union{Val{:SARSA},Val{:ExpectedSARSA},Val{:SARS}},
    t::Trajectory,
    ::PostEpisodeStage,
)
    S, A, R, T = [t[x] for x in SART]
    n, γ, Q = L.n, L.γ, L.approximator
    G = 0.0
    for i in 1:min(n + 1, length(R))
        G = R[end-i+1] + γ * G
        s, a = S[end-i], A[end-i]
        update!(Q, (s, a) => Q(s, a) - G)
        update!(L.Δ_approximator, G - Q(s, a))
    end
end

function _update!(
    L::DeltaLearner,
    ::Union{TabularQApproximator,LinearQApproximator},
    ::Val{:SARSA},
    t::Trajectory,
    ::PreActStage,
)
    S, A, R, T = [t[x] for x in SART]
    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n + 1
        s, a, s′, a′ = S[end-n-1], A[end-n-1], S[end], A[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * Q(s′, a′)
        update!(Q, (s, a) => Q(s, a) - G)
        update!(L.Δ_approximator, G - Q(s, a))
    end
end

function _update!(
    L::DeltaLearner,
    ::TabularQApproximator,
    ::Val{:ExpectedSARSA},
    experience,
    ::PreActStage,
)
    t, p = experience

    S = t[:state]
    A = t[:action]
    R = t[:reward]

    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n + 1
        s, a, s′ = S[end-n-1], A[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * dot(Q(s′), p)
        update!(Q, (s, a) => Q(s, a) - G)
        update!(L.Δ_approximator, G - Q(s, a))
    end
end

function _update!(
    L::DeltaLearner,
    ::Union{TabularQApproximator,LinearQApproximator},
    ::Val{:SARS},
    t::AbstractTrajectory,
    ::PreActStage,
)
    S = t[:state]
    A = t[:action]
    R = t[:reward]

    n, γ, Q = L.n, L.γ, L.approximator

    if length(R) >= n + 1
        s, a, s′ = S[end-n-1], A[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * maximum(Q(s′))
        update!(Q, (s, a) => Q(s, a) - G)
        update!(L.Δ_approximator, G - Q(s, a))
    end
end

function _update!(
    L::DeltaLearner,
    ::TabularBiasApproximator,
    ::Any,
    t::Trajectory,
    ::PreActStage,
)
end

function _update!(
    L::DeltaLearner,
    ::TabularBiasApproximator,
    ::Any,
    t::Trajectory,
    ::PostEpisodeStage,
)
    A_v = unique(t[:action])
    N = length(t[:action])
    Δ = L.Δ_approximator.Δ
    s = 1

    for a in A_v
        N_a = count(x -> x == a, t[:action])
        update!(L.offline_approximator, (s,a) => N_a * Δ / N)
    end
end

#---------
# Offline
#---------

function RLBase.update!(
    p::QBasedPolicy{<:DeltaLearner},
    m::ExperiencePrioritySamplingModel,
    ::AbstractTrajectory,
    env::AbstractEnv,
    ::PreActStage,
)   
end

function RLBase.update!(
    p::QBasedPolicy{<:DeltaLearner},
    m::ExperiencePrioritySamplingModel,
    ::AbstractTrajectory,
    env::AbstractEnv,
    ::PostEpisodeStage,
)   
    if p.learner.method == :SARS
        L = p.learner
        Δ = L.Δ_approximator.Δ
        transition = sample(m)

        if !isnothing(transition)
            s, a, r, d, s′, P = transition
            update!(L.offline_approximator, (s,a) => Δ)
        end
    else
        @error "unsupported method $(L.method)"
    end
end

function RLBase.priority(L::DeltaLearner, transition::Tuple)
    if L.method == :SARS
        s, a, r, d, s′ = transition
        γ, Q = L.γ, L.approximator
        δ = d ? (r - Q(s, a)) : (r + γ^(L.n + 1) * maximum(Q(s′)) - Q(s, a))
        δ = [δ]  # must be broadcastable in Flux.Optimise
        Flux.Optimise.apply!(Q.optimizer, (s, a), δ)
        abs(δ[])
    else
        @error "unsupported method"
    end
end