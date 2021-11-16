import StatsBase: sample
using StatsBase: Weights

Base.@kwdef mutable struct ExperiencePrioritySamplingModel{R} <: AbstractEnvironmentModel
    experiences::Vector{Tuple} = Vector{Tuple}()
    N_samples::Int
    rng::R = Random.GLOBAL_RNG
end

function RLBase.update!(
    m::ExperiencePrioritySamplingModel,
    t::AbstractTrajectory,
    p::AbstractPolicy,
    ::AbstractEnv,
    ::Union{PreActStage,PostEpisodeStage},
)
    if length(t[:terminal]) > 0
        transition = (
            t[:state][end-1],
            t[:action][end-1],
            t[:reward][end],
            t[:terminal][end],
            t[:state][end],
        )
        pri = RLBase.priority(p, transition)
        update!(m, (transition..., pri))
    end
end

function RLBase.update!(m::ExperiencePrioritySamplingModel, transition::Tuple)
    s, a, r, d, s′, P = transition
    push!(m.experiences, transition)
end

sample(model::ExperiencePrioritySamplingModel) = sample(model.rng, model)

function sample(rng::AbstractRNG, m::ExperiencePrioritySamplingModel)
    if length(m.experiences) > 0 
        w = Weights(map(x->x[end], m.experiences))
        sample(m.rng, m.experiences, w)
    else
        nothing
    end
end