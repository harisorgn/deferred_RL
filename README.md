# Deferred Reinforcement Learning

An investigation of reinforcement learning environments where rewards are delayed and might overlap with future actions, yet the causes are actions of the past. Temporal difference learning is a classic way to deal with such environments. Here we are proposing another learning rule, Delta learning, that splits its learning between an online and an offline phase, during and after each episode respectively. 

This is work in progress and a draft manuscript is being written on the Delta agent and how it could describe behavioural results from humans and animals, where subjects' affective state at one time influences learned experiences that occured within the same day, even if the affective state was different during the latter. This fact inspired the offline learning rule part of the Delta agent.

Environments and agents are implemented based on the ReinforcementLearning.jl package.
