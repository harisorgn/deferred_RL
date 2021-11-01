
using LinearAlgebra: dot
using Distributions: pdf

Base.@kwdef struct DeltaLearner <: AbstractLearner
    approximator::Union{TabularQApproximator,LinearQApproximator}
    Δ_approximator::DeltaApproximator
    bias_approximator::TabularBiasApproximator
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

(L::DeltaLearner)(env::AbstractEnv) = L.approximator(state(env)) + L.bias_approximator(state(env))
(L::DeltaLearner)(s) = L.approximator(s) + L.bias_approximator(s)
(L::DeltaLearner)(s, a) = L.approximator(s, a) + L.bias_approximator(s, a)

## update policies

function RLBase.update!(
    p::QBasedPolicy{<:DeltaLearner},
    t::AbstractTrajectory,
    e::AbstractEnv,
    s::AbstractStage,
)
    if p.learner.method === :ExpectedSARSA && s === PRE_ACT_STAGE
        # A special case
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
    _update!(L, L.bias_approximator, Val(L.method), t, s)
end

function RLBase.update!(
    L::DeltaLearner,
    t::AbstractTrajectory,
    ::AbstractEnv,
    s::PostEpisodeStage,
)
    _update!(L, L.approximator, Val(L.method), t, s)
    _update!(L, L.bias_approximator, Val(L.method), t, s)
end


# for ExpectedSARSA
function RLBase.update!(
    L::DeltaLearner,
    t::Tuple,
    ::AbstractEnv,
    s::Union{PreActStage,PostEpisodeStage},
)
    _update!(L, L.approximator, Val(L.method), t, s)
    _update!(L, L.bias_approximator, Val(L.method), t, s)
end

## update trajectories

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

## implementations

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
        update!(L.Δ_approximator, Q(s, a) - G)
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
        update!(L.Δ_approximator, Q(s, a) - G)
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
        update!(L.Δ_approximator, Q(s, a) - G)
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
        update!(L.Δ_approximator, Q(s, a) - G)
    end
end

function _update!(
    L::DeltaLearner,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::Val{:SRS},
    t::Trajectory,
    ::PostEpisodeStage,
)
    S, R = t[:state], t[:reward]
    n, γ, V = L.n, L.γ, L.approximator
    G = 0.0
    w = 1.0
    for i in 1:min(n + 1, length(R))
        G = R[end-i+1] + γ * G
        s = S[end-i]
        if haskey(t, :weight)
            w *= t[:weight][end-i]
        end
        update!(V, s => w * (V(s) - G))
        update!(L.Δ_approximator, w * (V(s) - G))
    end
end

function _update!(
    L::DeltaLearner,
    ::Union{TabularVApproximator,LinearVApproximator},
    ::Val{:SRS},
    t::AbstractTrajectory,
    ::PreActStage,
)
    S = t[:state]
    R = t[:reward]

    n, γ, V = L.n, L.γ, L.approximator
    if length(R) >= n + 1
        s, s′ = S[end-n-1], S[end]
        G = discount_rewards_reduced(@view(R[end-n:end]), γ) + γ^(n + 1) * V(s′)
        if haskey(t, :weight)
            W = t[:weight]
            @views w = reduce(*, W[end-n-1:end-1])
        else
            w = 1.0
        end
        update!(V, s => w * (V(s) - G))
        update!(L.Δ_approximator, w * (V(s) - G))
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
        update!(L.bias_approximator, (s,a) => N_a * Δ / N)
    end
end
