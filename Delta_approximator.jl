
Base.@kwdef mutable struct DeltaApproximator <: AbstractApproximator
    η::Float64
    Δ::Float64
end

DeltaApproximator(; η) = DeltaApproximator(η, 0.0)

(app::DeltaApproximator)(s::Int) = @view app.Δ[1]
(app::DeltaApproximator)(s::Int, a::Int) = @view app.Δ[1]

function RLBase.update!(app::DeltaApproximator, δ::Float64)
	app.Δ -= app.η*(δ + app.Δ)
end


