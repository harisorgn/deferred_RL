
Base.@kwdef mutable struct ARApproximator <: AbstractApproximator
    η::Float64
    ρ::Float64
end

ARApproximator(; η) = ARApproximator(η, 0.0)

function RLBase.update!(app::ARApproximator, G::Float64, Q::Float64, b::Float64)
	app.ρ += app.η*(G - app.ρ)
end


