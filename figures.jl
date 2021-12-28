using CairoMakie
using LaTeXStrings
using Serialization
using Statistics

function zero_intersect(R)

	n_episodes = size(R)[2]

	idx = findfirst.(x -> x>0.0, eachrow(R))
	idx[isnothing.(idx)] .= n_episodes
	median(idx)
end

final_reward(R) = mean(R[:,end])

function plot_sensitivity!(ax, f, R, H, agent_labels)

	@assert size(R)[4] == length(H)
	@assert size(R)[3] == length(agent_labels)

	n_agents = size(R)[3]

	for i = 1:n_agents

		t = mapslices(f, R[:,:,i,:]; dims=[1,2])
			
		lines!(ax, H, t[:], label=agent_labels[i], colormap=:seaborn_colorblind6)
	end
end

function plot_perform!(ax, R, agent_labels)

	@assert size(R)[3] == length(agent_labels)

	t = 1:size(R)[2] 

	n_agents = size(R)[3]

	for i = 1:n_agents

		μ = vec(mean(R[:,:,i], dims=1))
		σ = vec(std(R[:,:,i], dims=1))

		l = lines!(ax, t, μ, label=agent_labels[i], colormap = :seaborn_colorblind6)
		#band!(ax, t, μ+σ, μ-σ, color=(l.color,0.5), colormap = :seaborn_colorblind6)
	end
end

function plot_figure(R, H, agent_labels, env_labels)

	n_environments = length(env_labels)

	H_env = H[1:n_environments:end]

	idx_perform = 1

	f = Figure(resolution = (1280, 960))
	ax = [
			[	
			Axis(f[1,i], xlabel="Episode", ylabel=L"\tilde{R}", title=env_labels[i]),
			Axis(f[2,i], xlabel="learning rate", ylabel=L"T_0"), 
			Axis(f[3,i], xlabel="learning rate", ylabel=L"R_e")
			]
		for i=1:n_environments
		]

	for i=1:n_environments

		R_env = R[:,:,:,i:n_environments:end]

		plot_perform!(ax[i][1], R_env[:,:,:,idx_perform], agent_labels)
		plot_sensitivity!(ax[i][2], zero_intersect, R_env, H_env, agent_labels)
		plot_sensitivity!(ax[i][3], final_reward, R_env, H_env, agent_labels)
	end

	f[1, n_environments+1] = Legend(f, ax[1][1], framevisible=false, unique=true)

	map(x -> hideydecorations!.(x, grid = false), ax[2:end])

	return f
end


R = deserialize("R_deferred.jls")
H = deserialize("H_deferred.jls")
agent_labels = ["Delta", "Delta, online", "MC"]
env_labels = [L"DB_1", L"DB_2", L"DB_3"]

f = plot_figure(R, H, agent_labels, env_labels)

save("res.eps", f)

