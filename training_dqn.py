import gymnasium
import highway_env
import numpy as np
import torch
from torch import nn
import random

from implementations_dqn import DQN, DDQNAgent

import os


MAX_STEPS = int(3e4)    # Number of steps performed
DISCOUNT_FACTOR = 0.99  # Discount factor for future rewards, can be changed

BUFFER_SIZE = 10000
BATCH_SIZE = 128
ALPHA = 1e-3
EPSILON = 1.0
EPSILON_DECAY_RATE = 0.995
EPSILON_MIN = 0.01
TARGET_UPDATE_FREQ = 1000

WINDOW_SIZE = 100  # Size of the window for calculating the average loss and return
N_STEPS = 400  # Number of steps after which the average loss is printed


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


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

# ---------------------------------------- Check the method to be used ----------------------------------------

dqn_type = ''
print("Which type of DQN would you like to train? (1 for Double DQN, 2 for Dueling DQN)")
while dqn_type not in ['1', '2']:
    dqn_type = input("Enter your choice (1 or 2): ")
    if dqn_type not in ['1', '2']:
        print("Invalid choice. Please enter 1 or 2.")

dqn_name = "Double_DQN" if dqn_type == '1' else "Dueling_DQN" if dqn_type == '2' else "Unknown_DQN_Type"

if dqn_name == "Double_DQN":
    print("Training Double DQN...")
    q_network_1 = DQN(state_dim, num_actions)
    target_network = DQN(state_dim, num_actions)

    # Initialize weights
    q_network_1.apply(q_network_1.init_weights)
    target_network.apply(target_network.init_weights)

    agent = DDQNAgent(env,
                      q_network_1=q_network_1,
                      target_network=target_network,
                      discount_factor=DISCOUNT_FACTOR,
                      buffer_size=BUFFER_SIZE,
                      batch_size=BATCH_SIZE,
                      alpha=ALPHA,
                      epsilon=EPSILON,
                      epsilon_decay_rate=EPSILON_DECAY_RATE,
                      epsilon_min=EPSILON_MIN,
                      target_update_freq=TARGET_UPDATE_FREQ)

elif dqn_name == "Dueling_DQN":
    print("Training Dueling DQN...")
    q_network_1 = DQN(state_dim, num_actions, enable_dueling_dqn=True)
    target_network = DQN(state_dim, num_actions, enable_dueling_dqn=True)

    # Initialize weights
    q_network_1.apply(q_network_1.init_weights)
    target_network.apply(target_network.init_weights)

    agent = DDQNAgent(env,
                      q_network_1=q_network_1,
                      target_network=target_network,
                      discount_factor=DISCOUNT_FACTOR,
                      buffer_size=BUFFER_SIZE,
                      batch_size=BATCH_SIZE,
                      alpha=ALPHA,
                      epsilon=EPSILON,
                      epsilon_decay_rate=EPSILON_DECAY_RATE,
                      epsilon_min=EPSILON_MIN,
                      target_update_freq=TARGET_UPDATE_FREQ)

else:
    raise NotImplementedError("DQN type not implemented/recognized.")

# -----------------------------------------------------------------------------------------------------------




episode = 1
episode_steps = 0
episode_return = 0

episode_returns = []
episode_lengths = []

state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

losses = []  # List to store losses for each training step
loss_window = []  # List to store losses for the last N training steps
mean_losses = []
mean_returns = []

# Training loop
for t in range(MAX_STEPS):
    episode_steps += 1

    # Select the action to be performed by the agent
    action = agent.select_action(state)

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    # Store transition in memory and train your model
    agent.store_transition(state, action, float(reward), next_state, done)

    loss = agent.train()
    if loss is not None:
        losses.append(loss)  # Store the loss for this training step
        loss_window.append(loss)

        if len(loss_window) > WINDOW_SIZE:
            loss_window.pop(0)


    # Print the average loss every N_STEPS
    if (t + 1) % N_STEPS == 0 and len(loss_window) > 0:
        mean_loss = sum(loss_window) / len(loss_window)
        current_loss = loss if loss is not None else "N/A"
        print(f"Step {t+1:5d}/{MAX_STEPS} | "
            f"Mean Loss: {mean_loss:7.4f} | "
            f"Current: {current_loss:>7} | "
            f"ε: {agent.epsilon:.3f} | "
            f"Episode: {episode}")
        
        mean_losses.append(mean_loss)
    # -----------------------------------

    state = next_state
    episode_return += float(reward)

    if done or truncated:
        print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Decay epsilon
        agent.epsilon_decay()

        # Save training information and model parameters
        episode_returns.append(episode_return)
        episode_lengths.append(episode_steps)

        means = [episode_returns[i] for i in range(max(0, episode - WINDOW_SIZE), episode)]
        print("Mean of episode returns: ", np.mean(means))
        mean_returns.append(np.mean(means))

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0



# Save the final model parameters

save_dir = f"results_{dqn_name}"
os.makedirs(save_dir, exist_ok=True)

agent.save(os.path.join(save_dir, f"ddqn_agent_final_{dqn_name}.pth"))
np.save(os.path.join(save_dir, f"episode_returns_final_{dqn_name}.npy"), np.array(episode_returns))
np.save(os.path.join(save_dir, f"episode_lengths_final_{dqn_name}.npy"), np.array(episode_lengths))
np.save(os.path.join(save_dir, f"losses_final_{dqn_name}.npy"), np.array(losses))

env.close()


# Plotting the mean losses and the returns
agent.plot_mean_losses(mean_losses, dqn_name, save_the_plot=True, save_dir=save_dir)
agent.plot_returns(episode_returns, dqn_name, save_the_plot=True, save_dir=save_dir)
agent.plot_mean_returns(mean_returns, dqn_name, save_the_plot=True, save_dir=save_dir)