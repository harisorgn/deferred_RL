
struct TabularBiasApproximator{N,T<:AbstractArray{<:AbstractFloat, N}} <: AbstractApproximator
    table::T
    η::Float64
    function TabularBiasApproximator(table::T, η::Float64) where {T<:AbstractArray}
        n = ndims(table)
        n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
        new{n,T}(table, η)
    end
end

TabularBiasApproximator(; n_state, n_action, init = 0.0, η) =
    TabularBiasApproximator(fill(init, n_action, n_state), η)

(app::TabularBiasApproximator)(s::Int) = @views app.table[:, s]
(app::TabularBiasApproximator)(s::Int, a::Int) = app.table[a, s]

function RLBase.update!(app::TabularBiasApproximator, correction::Pair{Tuple{Int,Int},Float64})
    (s, a), e = correction
    app.table[a, s] += app.η * e
end
