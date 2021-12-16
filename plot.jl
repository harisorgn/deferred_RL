using CairoMakie

function plot_perform!(R, ax, label::String)

	t = 1:size(R)[2] 

	μ = vec(mean(R, dims=1))
	σ = vec(std(R, dims=1))

	l = lines!(ax, t, μ, label=label, colormap = :seaborn_colorblind6)

	band!(ax, t, μ+σ, μ-σ, color=(l.color,0.5), colormap = :seaborn_colorblind6)
end

function plot_agents(R, agent_labels, env_labels)

	@assert length(R) == length(env_labels)
	@assert length(R[1]) == length(agent_labels)

	f = Figure()

	ax = [
	    Axis(f[i,1], xlabel="Episode", ylabel=L"\tilde{R}", title=env_labels[i])
	    for i=1:length(env_labels)
	    ]

	map(zip(R, ax)) do (R_env, ax_env)

	    plot_perform!.(
	                    R_env, 
	                    ax_env, 
	                    agent_labels 
	                    )

	end

	hidexdecorations!.(ax[1:end-1])
	f[1, 2] = Legend(f, ax[1], framevisible = false)

	save("res.eps", f)
end