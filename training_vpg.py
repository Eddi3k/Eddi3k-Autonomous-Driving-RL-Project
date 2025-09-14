import os
import gymnasium
import highway_env
import numpy as np
import torch
import random
import implementation_vpg as vpg


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "vehicles_count": 50})


# Check if the observation and action spaces are as expected
if not isinstance(env.observation_space, gymnasium.spaces.Box):
    raise ValueError("Expected Box observation space")

if not isinstance(env.action_space, gymnasium.spaces.Discrete):
    raise ValueError("Expected Discrete action space")

# Extract dimensions
if not len(env.observation_space.shape) == 2:
    raise ValueError("Expected 2D observation space, got {}".format(env.observation_space.shape))
state_dim = int(env.observation_space.shape[0] * env.observation_space.shape[1])  # Assuming the observation space is a 2D Box
num_actions = int(env.action_space.n)


print("Training VPG...")

# Initialize your model
agent = vpg.Vanilla_Policy_Gradient(vpg.Network(state_dim, num_actions))

state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0


episode_rewards = []  # Store rewards per timestep
log_prob_actions = []   # Store log probabilities of actions

returns = []            # Save all the obtained returns

# Training loop
for t in range(MAX_STEPS):
    episode_steps += 1

    policy = agent.get_policy(agent.model, state)

    # Select the action to be performed by the agent
    action, log_prob_action = agent.select_action(policy)

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    # Store transition in memory and train your model
    episode_rewards.append(reward)
    log_prob_actions.append(log_prob_action)

    state = next_state
    episode_return += float(reward)

    if done or truncated:
        print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Save training information and model parameters

        returns_to_go = []
        running_return = 0
        for reward in reversed(episode_rewards):
            running_return += reward
            returns_to_go.insert(0, running_return)

        # Normalize the returns to go
        returns_to_go = torch.tensor(returns_to_go, dtype=torch.float32)

        returns.append(episode_return)

        step_loss = agent.calculate_loss(torch.stack(log_prob_actions), returns_to_go)

        agent.optimizer.zero_grad() # Clear gradients from previous step
        step_loss.backward()    # Compute gradients
        agent.optimizer.step()  # Update parameters

        log_prob_actions.clear()
        episode_rewards.clear()

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

# Save training results and plot stuff
save_dir = "results_VPG"
os.makedirs(save_dir, exist_ok=True)

agent.save(os.path.join(save_dir, f"drl_agent_final_VPG.pth"))
agent.plot_returns(returns, save_the_plot=True, save_dir=save_dir)