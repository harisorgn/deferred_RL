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

function plot_perform!(ax, R, idx_perform, agent_labels)

	@assert size(R)[3] == length(agent_labels)

	t = 1:size(R)[2] 

	n_agents = size(R)[3]

	for i = 1:n_agents

		μ = vec(mean(R[:,:,i, idx_perform[i]], dims=1))
		σ = vec(std(R[:,:,i, idx_perform[i]], dims=1))

		lines!(ax, t, μ, label=agent_labels[i], colormap = :seaborn_colorblind6)
		#band!(ax, t, μ+σ, μ-σ, color=(l.color,0.5), colormap = :seaborn_colorblind6)
	end
end

function plot_figure(R, H, agent_labels, env_labels)

	fig_sz_inch = (6.4, 8)
	font_sz = 12

	n_environments = length(env_labels)

	H_env = H[1:n_environments:end]

	idx_perform = [1, 1, findfirst(x -> x==0.55, H_env)]

	f = Figure(resolution = 72 .* fig_sz_inch, fontsize=font_sz)

	ax = [
			[	
			Axis(f[1,i], xlabel="Episode", ylabel=L"\tilde{R}", 
						title=env_labels[i], titlesize=26),
			Axis(f[2,i], xlabel="learning rate", ylabel=L"T_0"), 
			Axis(f[3,i], xlabel="learning rate", ylabel=L"R_e")
			]
		for i=1:n_environments
		]

	for i=1:n_environments

		R_env = R[:,:,:,i:n_environments:end]

		plot_perform!(ax[i][1], R_env, idx_perform, agent_labels)
		plot_sensitivity!(ax[i][2], zero_intersect, R_env, H_env, agent_labels)
		plot_sensitivity!(ax[i][3], final_reward, R_env, H_env, agent_labels)
	end

	f[1, n_environments+1] = Legend(f, ax[1][1], framevisible=false, unique=true)

	map(x -> hideydecorations!.(x, grid=false, ticklabels=false, ticks=false), ax[2:end])

	return f
end

function example_performance(R, H, agent_labels, env_labels)

	fig_sz_inch = (6.4,8)
	font_sz = 12

	n_environments = length(env_labels)

	H_env = H[1:n_environments:end]

	idx_perform = [1, 1, findfirst(x -> x==0.55, H_env)]

	f = Figure(resolution = 72 .* fig_sz_inch, fontsize=font_sz)

	ga = f[1, 1] = GridLayout()
	gb = f[2, 1] = GridLayout()
	gc = f[3, 1] = GridLayout()
	gd = f[4, 1] = GridLayout()

	ax = [
		Axis(ga[1,1], 
			xlabel="Episode", 
			ylabel=L"\tilde{R}", 
			title=env_labels[1], 
			titlesize=font_sz),
		Axis(gb[1,1], 
			xlabel="Episode", 
			ylabel=L"\tilde{R}", 
			title=env_labels[2], 
			titlesize=font_sz),
		Axis(gc[1,1], 
			xlabel="Episode", 
			ylabel=L"\tilde{R}", 
			title=env_labels[3], 
			titlesize=font_sz),
		Axis(gd[1,1], 
			xlabel="Episode", 
			ylabel=L"\tilde{R}", 
			title=env_labels[4], 
			titlesize=font_sz)
		]

	for i = 1:n_environments
		R_env = R[:,:,:,i:n_environments:end]
		plot_perform!(ax[i], R_env, idx_perform, agent_labels)
	end

	hidexdecorations!(ax[1], grid=false)
	hidexdecorations!(ax[2], grid=false)
	hidexdecorations!(ax[3], grid=false)

	for (label, layout) in zip(["A", "B", "C", "D"], [ga, gb, gc, gd])
	    Label(layout[1, 1, TopLeft()], label,
	        textsize = 18,
	        halign = :right)
	end

	Legend(f[5,1], ax[1], framevisible=true, orientation=:horizontal, tellwidth=false, tellheight=true, labelsize=9)

	rowgap!(f.layout, Relative(0.02))

	save("perform.eps", f, pt_per_unit=1)
end

function T_sensitivity(R, H, agent_labels, env_labels)

	fig_sz_inch = (6.4,8)
	font_sz = 12

	n_environments = length(env_labels)

	H_env = H[1:n_environments:end]

	idx_perform = [1, 1, findfirst(x -> x==0.55, H_env)]

	f = Figure(resolution = 72 .* fig_sz_inch, fontsize=font_sz)

	ga = f[1, 1] = GridLayout()
	gb = f[2, 1] = GridLayout()
	gc = f[3, 1] = GridLayout()
	gd = f[4, 1] = GridLayout()

	ax = [
		Axis(ga[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"T_0",
			title=env_labels[1], 
			titlesize=font_sz),
		Axis(gb[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"T_0", 
			title=env_labels[2], 
			titlesize=font_sz),
		Axis(gc[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"T_0",  
			title=env_labels[3], 
			titlesize=font_sz),
		Axis(gd[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"T_0",  
			title=env_labels[4], 
			titlesize=font_sz)
		]

	for i = 1:n_environments
		R_env = R[:,:,:,i:n_environments:end]
		plot_sensitivity!(ax[i], zero_intersect, R_env, H_env, agent_labels)
	end

	hidexdecorations!(ax[1], grid=false)
	hidexdecorations!(ax[2], grid=false)
	hidexdecorations!(ax[3], grid=false)

	for (label, layout) in zip(["A", "B", "C", "D"], [ga, gb, gc, gd])
	    Label(layout[1, 1, TopLeft()], label,
	        textsize = 18,
	        halign = :right)
	end

	Legend(f[5,1], ax[1], framevisible=true, orientation=:horizontal, tellwidth=false, tellheight=true, labelsize=9)

	rowgap!(f.layout, Relative(0.02))

	save("T_0.eps", f, pt_per_unit=1)
end

function R_sensitivity(R, H, agent_labels, env_labels)

	fig_sz_inch = (6.4,8)
	font_sz = 12

	n_environments = length(env_labels)

	H_env = H[1:n_environments:end]

	idx_perform = [1, 1, findfirst(x -> x==0.55, H_env)]

	f = Figure(resolution = 72 .* fig_sz_inch, fontsize=font_sz)

	ga = f[1, 1] = GridLayout()
	gb = f[2, 1] = GridLayout()
	gc = f[3, 1] = GridLayout()
	gd = f[4, 1] = GridLayout()

	ax = [
		Axis(ga[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"R_e",
			title=env_labels[1], 
			titlesize=font_sz),
		Axis(gb[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"R_e", 
			title=env_labels[2], 
			titlesize=font_sz),
		Axis(gc[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"R_e",  
			title=env_labels[3], 
			titlesize=font_sz),
		Axis(gd[1,1], 
			xlabel=latexstring("Online learning rate, \$ \\eta \$"), 
			ylabel=L"R_e",  
			title=env_labels[4], 
			titlesize=font_sz)
		]

	for i = 1:n_environments
		R_env = R[:,:,:,i:n_environments:end]
		plot_sensitivity!(ax[i], final_reward, R_env, H_env, agent_labels)
	end

	hidexdecorations!(ax[1], grid=false)
	hidexdecorations!(ax[2], grid=false)
	hidexdecorations!(ax[3], grid=false)

	for (label, layout) in zip(["A", "B", "C", "D"], [ga, gb, gc, gd])
	    Label(layout[1, 1, TopLeft()], label,
	        textsize = 18,
	        halign = :right)
	end
	
	Legend(f[5,1], ax[1], framevisible=true, orientation=:horizontal, tellwidth=false, tellheight=true, labelsize=9)

	rowgap!(f.layout, Relative(0.02))

	save("R_e.eps", f, pt_per_unit=1)
end

R = deserialize("R.jls")
H = deserialize("H.jls")

agent_labels = ["Delta", "Delta-online", "MC"]
env_labels = [
			"DB₁ : no rewards",
			"DB₂ : binary rewards",
			"DB₃ : continuous rewards",
			"TB"
			]

example_performance(R, H, agent_labels, env_labels)
T_sensitivity(R, H, agent_labels, env_labels)
R_sensitivity(R, H, agent_labels, env_labels)
