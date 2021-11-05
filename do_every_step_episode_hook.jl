mutable struct DoEveryNStepEveryEpisode{F} <: AbstractHook
    f::F
    n::Int
    t::Int
end

DoEveryNStepEveryEpisode(f; n = 1, t = 0) = DoEveryNStepEveryEpisode(f, n, t)

function (hook::DoEveryNStepEveryEpisode)(::PostActStage, agent, env)
    hook.t += 1
    if hook.t % hook.n == 0
        hook.f(hook.t, agent, env)
        hook.t = 0
    end
end
