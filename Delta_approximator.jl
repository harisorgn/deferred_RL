
Base.@kwdef mutable struct DeltaApproximator <: AbstractApproximator
    η::Float64
    Δ::Float64
end

DeltaApproximator(; η) = DeltaApproximator(η, 0.0)

function RLBase.update!(app::DeltaApproximator, G::Float64, Q::Float64, b::Float64)
	app.Δ += app.η*(G - Q - b - app.Δ)
end


