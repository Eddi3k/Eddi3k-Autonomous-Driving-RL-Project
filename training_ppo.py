import gymnasium
import highway_env
import numpy as np
import torch
from torch import nn
import random

from implementation_ppo_clip import ACNetwork, PPO

import os


MAX_STEPS = int(4e4)    # Number of steps performed
BATCH_SIZE = 4000    # Number of episodes to run per batch (== foreach rollout)
MAX_ROLLOUTS = MAX_STEPS // BATCH_SIZE    # Number of steps performed
if MAX_ROLLOUTS == 0:
    print("Warning: MAX_ROLLOUTS is 0! Consider increasing MAX_STEPS or decreasing BATCH_SIZE.")
    
DISCOUNT_FACTOR = 0.99  # Discount factor for future rewards, can be changed
PPO_STEPS = 4   # Number of times each batch is re-used to run the backward pass
EPSILON = 0.2
ENTROPY_COEFFICIENT = 0.01
CLIP_THRESHOLD = 0.2
MAX_GRAD_NORM = 0.5  # Gradient norm clipping value
TARGET_KL = 0.02  # KL divergence target value for early stopping

WINDOW_SIZE = 100


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


print("Training PPO...")


actor = ACNetwork(state_dim, num_actions)
critic = ACNetwork(state_dim, 1)   # Since the critic network predicts just the expected value (given an input state), the number of output features is 1.

# Initialize weights
actor.apply(actor.init_weights)
critic.apply(critic.init_weights)

agent = PPO(env=env, actor=actor, critic=critic)

episode = 1
episode_steps = 0
episode_return = 0

episode_returns = []    # Used to store the return after each trajectory
episode_lengths = []

# For performance monitoring
current_actor_losses = []
current_critic_losses = []

# To store the mean actor and critic losses for each episode
total_actor_losses = []
total_critic_losses = []

mean_returns = []

k = 0
# Training loop
while k < MAX_ROLLOUTS:

    # Rollout to collect set of trajectories
    batch_states, batch_actions, batch_log_probabilities, batch_rewards, episode, k, episode_returns, episode_lengths = agent.rollout(BATCH_SIZE, episode, k, episode_returns, episode_lengths)

    # Calculate rewards_to_go
    batch_rewards_to_go = agent.calculate_rewards_to_go(batch_rewards, discount_factor=DISCOUNT_FACTOR)

    # Calculate the value predictions to use for advantage estimation
    V = agent.evaluate(batch_states)
    # Calculate advantage estimetion at t-th iteration (and normalize them)
    A_k = agent.calculate_advantages(batch_rewards_to_go, V)

    # Learning rate annealing
    print("Previous learning rates: ", agent.actor_optimizer.param_groups[0]["lr"], agent.critic_optimizer.param_groups[0]["lr"])
    frac = (k - 1.0) / MAX_STEPS
    new_lr_actor = agent.actor_optimizer.param_groups[0]["lr"] * (1.0 - frac)
    new_lr_actor = max(new_lr_actor, 0)
    new_lr_critic = agent.critic_optimizer.param_groups[0]["lr"] * (1.0 - frac)
    new_lr_critic = max(new_lr_critic, 0)
    agent.actor_optimizer.param_groups[0]["lr"] = new_lr_actor
    agent.critic_optimizer.param_groups[0]["lr"] = new_lr_critic
    print("Updated learning rates: ", agent.actor_optimizer.param_groups[0]["lr"], agent.critic_optimizer.param_groups[0]["lr"])


    for _ in range(PPO_STEPS):

        # Update policy maximizing ppo-clip objective (== update actor parameters)
        # Compute the value predictions
        V = agent.evaluate(batch_states)
        # Compute the new log probabilities
        current_log_probabilities = agent.get_policy(batch_states).log_prob(batch_actions)
        # Compute the surrogate loss and get the policy ratios for KL divergence monitoring
        policy_log_ratios, policy_ratios, surrogate_loss_1, surrogate_loss_2 = agent.calculate_surrogate_losses(batch_log_probabilities.detach(), current_log_probabilities, CLIP_THRESHOLD, A_k)
        # Compute the actor loss
        actor_loss = (-torch.min(surrogate_loss_1, surrogate_loss_2)).mean()
        # Apply the entropy regularization
        entropy_loss = agent.get_policy(batch_states).entropy().mean()
        actor_loss -= ENTROPY_COEFFICIENT * entropy_loss
        # Compute the gradients and perform backward propagation for actor network
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), MAX_GRAD_NORM)  # Gradient clipping
        agent.actor_optimizer.step()


        # Fit value function (== update critic parameters)
        critic_loss = nn.MSELoss()(V, batch_rewards_to_go)
        # Compute the gradients and perform backward propagation for critic network
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.critic.parameters(), MAX_GRAD_NORM)  # Gradient clipping
        agent.critic_optimizer.step()

        # Compute the approximate KL divergence for monitoring
        approx_kl = ((policy_ratios - 1) - policy_log_ratios).mean()
        # Break in case of large policy update
        if approx_kl > TARGET_KL:
            print("Early stopping due to reaching max kl.")
            break

        current_actor_losses.append(actor_loss.item())
        current_critic_losses.append(critic_loss.item())

    # Performance monitoring
    avg_actor_loss = np.mean(current_actor_losses)  # mean of the last "PPO_STEPS" actor losses
    avg_critic_loss = np.mean(current_critic_losses)  # mean of the last "PPO_STEPS" critic losses

    total_actor_losses.append(avg_actor_loss)
    total_critic_losses.append(avg_critic_loss)

    print(f"Episode: {episode} | Average Actor Loss: {avg_actor_loss:7.4f} | Average Critic Loss: {avg_critic_loss:7.4f}")

    current_actor_losses.clear()
    current_critic_losses.clear()


    means = [episode_returns[i] for i in range(max(0, len(episode_returns) - WINDOW_SIZE), len(episode_returns))]
    print("Mean of episode returns: ", np.mean(means))
    mean_returns.append(np.mean(means))

env.close()

# Save the final model parameters

save_dir = "results_PPO"
os.makedirs(save_dir, exist_ok=True)

agent.save(os.path.join(save_dir, f"drl_agent_final_PPO.pth"))
np.save(os.path.join(save_dir, f"episode_returns_final_PPO.npy"), np.array(episode_returns))
np.save(os.path.join(save_dir, f"episode_lengths_final_PPO.npy"), np.array(episode_lengths))
np.save(os.path.join(save_dir, f"actor_losses_final_PPO.npy"), np.array(total_actor_losses))
np.save(os.path.join(save_dir, f"critic_losses_final_PPO.npy"), np.array(total_critic_losses))

# Plot results
agent.plot_returns(episode_returns, save_the_plot=True, save_dir=save_dir)
agent.plot_mean_returns(mean_returns, save_the_plot=True, save_dir=save_dir)
agent.plot_actor_losses(total_actor_losses, save_the_plot=True, save_dir=save_dir)
agent.plot_critic_losses(total_critic_losses, save_the_plot=True, save_dir=save_dir)