using CairoMakie: lines!, band!, save, Figure, Axis

function plot_band(r_Delta, r_mc, r_no_Delta)
	f = Figure()
	Axis(f[1, 1])

	t = 1:n_episodes

	μ_Delta = vec(mean(r_Delta, dims=1))
	σ_Delta = vec(std(r_Delta, dims=1))

	μ_mc = vec(mean(r_mc, dims=1))
	σ_mc = vec(std(r_mc, dims=1))

	μ_no = vec(mean(r_no_Delta, dims=1))
	σ_no = vec(std(r_no_Delta, dims=1))

	lines!(t, μ_Delta, color=:blue)
	lines!(t, μ_mc, color=:red)
	lines!(t, μ_no, color=:green)

	band!(t, μ_Delta + σ_Delta, μ_Delta - σ_Delta, color=(:blue,0.5))
	band!(t, μ_mc + σ_mc, μ_mc - σ_mc, color=(:red,0.5))
	band!(t, μ_no + σ_no, μ_no - σ_no, color=(:green,0.2))

	save("res.png", f)
end