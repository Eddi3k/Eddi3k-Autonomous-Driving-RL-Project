import os
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import matplotlib.pyplot as plt

NUMBER_HIDDEN_FEATURES = 128
STARTING_LEARNING_RATE_ACTOR = 3e-4
STARTING_LEARNING_RATE_CRITIC = 3e-4

class ACNetwork(nn.Module):
    def __init__(self, number_observation_features: int, number_actions: int):
        super(ACNetwork, self).__init__()
        self.fc1 = nn.Linear(number_observation_features, NUMBER_HIDDEN_FEATURES)
        self.fc2 = nn.Linear(NUMBER_HIDDEN_FEATURES, NUMBER_HIDDEN_FEATURES)
        self.fc3 = nn.Linear(NUMBER_HIDDEN_FEATURES, number_actions)

    def forward(self, observation):
        # Convert observation to tensor if it's a numpy array
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float)

        activation1 = F.relu(self.fc1(observation))
        activation2 = F.relu(self.fc2(activation1))
        output = self.fc3(activation2)
        return output
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    

class PPO:
    def __init__(self, env: gymnasium.Env, actor: ACNetwork, critic: ACNetwork):
        super().__init__()
        self.actor = actor
        self.critic = critic

        self.env = env

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=STARTING_LEARNING_RATE_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=STARTING_LEARNING_RATE_CRITIC)


    def get_policy(self, observations: torch.Tensor) -> distributions.Categorical:
        '''Returns the policy distribution for a given observation.

        Args:
            observations (torch.Tensor): The observations (== states) for which to compute the policy.

        Returns:
            distributions.Categorical: Multinomial distribution parameterized by actor logits.
        '''

        logits = self.actor(observations)

        # Categorical will also normalize the logits for us
        return distributions.Categorical(logits=logits)
    

    def select_action(self, policy: distributions.Categorical):
        '''Samples an action from the given policy distribution.'''

        action = policy.sample()
        log_probability_action = policy.log_prob(action)    # We keep track of log probabilities instead of raw action probabilities as it makes gradient ascent easier behind the scenes

        return action.detach().numpy(), log_probability_action.detach()


    def rollout(self, timesteps_per_batch: int, episode: int, current_total_timesteps: int, episode_returns: list, episode_lengths: list):
        '''Perform a rollout (simulation) of the current policy in the environment.\n
        In particular, a rollout is a sequence of interactions between agent and environment collected under the current policy (not the optimal one).
        
        Args:
            timesteps_per_batch (int): Number of steps to run per batch (== foreach rollout).
            episode (int): Current episode number.
            current_total_timesteps (int): Current total number of timesteps performed.
            episode_returns (list): List containing the return obtained after each trajectory.
            episode_lengths (list): List containing the length of each trajectory.

        '''

        # Batch data
        batch_states = []
        batch_actions = []
        batch_log_probabilities = []
        batch_rewards = []

        episode_rewards = []        # Store rewards per timestep
        episode_return = 0
        episode_steps = 0

        batch_state, _ = self.env.reset()
        batch_state = batch_state.reshape(-1)
        done, truncated = False, False

        for _ in range(timesteps_per_batch):

            current_total_timesteps += 1
            episode_steps += 1
            policy = self.get_policy(batch_state)

            # Select the action to be performed by the agent
            batch_action, log_prob = self.select_action(policy)

            # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
            next_batch_state, reward, done, truncated, _ = self.env.step(batch_action)
            next_batch_state = next_batch_state.reshape(-1)

            # Store transition in memory and train your model
            batch_states.append(batch_state)
            batch_actions.append(batch_action)
            batch_log_probabilities.append(log_prob)
            episode_rewards.append(reward)

            batch_state = next_batch_state
            episode_return += float(reward)

            if done or truncated:        
                print(f"Total T: {current_total_timesteps} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")
                # Save training information and model parameters
                episode_returns.append(episode_return)
                episode_lengths.append(episode_steps)
                batch_rewards.append(episode_rewards)

                batch_state, _ = self.env.reset()
                batch_state = batch_state.reshape(-1)
                episode += 1
                episode_return = 0
                episode_steps = 0
                episode_rewards = []

                # Reset values after each episode, as one of them became True
                done, truncated = False, False

        # Manage truncated episodes due to the exit from the loop saving the rewards obtained so far in the last (truncated) episode
        if episode_rewards:
            batch_rewards.append(episode_rewards)
            episode_returns.append(episode_return)
            episode_lengths.append(episode_steps)

        # Reshape data to tensor
        batch_states = torch.from_numpy(np.array(batch_states)).float()
        batch_actions = torch.from_numpy(np.array(batch_actions)).float()
        batch_log_probabilities = torch.tensor(batch_log_probabilities, dtype=torch.float)

        return batch_states, batch_actions, batch_log_probabilities, batch_rewards, episode, current_total_timesteps, episode_returns, episode_lengths


    def calculate_rewards_to_go(self, batch_rewards, discount_factor):
        '''Computes the rewards-to-go per episode per batch.\n
            Rewards-to-go are the cumulative rewards from each timestep to the end of the episode.

            Args:
                batch_rewards (list): List of lists of rewards obtained in the batch, one list per episode.
                discount_factor (float): Discount factor (gamma) to be used in the computation of rewards-to-go.
        '''

        batch_rewards_to_go = []

        for episode_rewards in reversed(batch_rewards):
            cumulative_reward = 0   # The discounted reward so far

            # Go backwards for efficiency
            for reward in reversed(episode_rewards):
                cumulative_reward = reward + cumulative_reward * discount_factor
                batch_rewards_to_go.insert(0, cumulative_reward)

        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)

        # Normalize the rewards-to-go to manage the different numbers of timesteps per batch
        batch_rewards_to_go = (batch_rewards_to_go - batch_rewards_to_go.mean()) / (batch_rewards_to_go.std() + 1e-10)

        return batch_rewards_to_go
    
    
    def calculate_advantages(self, batch_rewards_to_go, values):
        '''Calculate the advantages.
        
        Args:
            batch_rewards_to_go (torch.Tensor): The batch rewards-to-go from the rollouts.
            values (torch.Tensor): The value function estimates (== V).

        '''

        advantages = batch_rewards_to_go - values.detach()

        # Safely normalize advantages with proper handling of edge cases. (WRITE IN REPORT)
        adv_mean = advantages.mean()
        # Use unbiased=False to avoid the degrees of freedom issue (WRITE IN REPORT)
        adv_std = torch.std(advantages, unbiased=False)
        
        if adv_std > 1e-10:
            return (advantages - adv_mean) / adv_std
        else:
            # If std is too small, just center the advantages
            return advantages - adv_mean
        

    def calculate_surrogate_losses(self, actions_log_probability_old, actions_log_probability_new, clip_threshold, advantages):
        '''Compute the surrogate loss for PPO using clip'''

        advantages = advantages.detach()
        policy_log_ratios = actions_log_probability_new - actions_log_probability_old
        policy_ratios = (policy_log_ratios).exp()

        # Not-clipped surrogate loss
        surrogate_loss_1 = policy_ratios * advantages

        # Clipped surrogate loss
        surrogate_loss_2 = torch.clamp(policy_ratios, min=1.0-clip_threshold, max=1.0+clip_threshold) * advantages
        
        return policy_log_ratios, policy_ratios, surrogate_loss_1, surrogate_loss_2
    

    def evaluate(self, batch_states):
        '''Evaluate the value function for a batch of states.'''

        V = self.critic(batch_states)
        return V.squeeze()


    def save(self, file_path: str):
        """Save the agent's parameters to a file"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, file_path)
        print(f"Agent parameters saved to {file_path}")


    # ------------------------------------ Plot results ------------------------------------

    def plot_returns(self, episode_returns: list, save_the_plot: bool, save_dir):
        """Plot the episode returns"""
        plt.plot(range(len(episode_returns)), episode_returns)
        plt.xlabel('Episode')
        plt.ylabel('Episode Return')
        plt.title('Episode Returns through episodes')
        if save_the_plot: plt.savefig(os.path.join(save_dir, f"episode_returns_final_PPO.png"))
        plt.show()

    def plot_mean_returns(self, mean_returns: list, save_the_plot: bool, save_dir):
        """Plot the average episode returns, taken considering windows of  "WINDOW_SIZE" steps"""
        plt.plot(range(len(mean_returns)), mean_returns)
        plt.xlabel('Episode')
        plt.ylabel('Average Episode Return')
        plt.title('Average Episode Returns through episodes')
        if save_the_plot: plt.savefig(os.path.join(save_dir, f"mean_episode_returns_final_PPO.png"))
        plt.show()

    def plot_actor_losses(self, total_actor_losses: list, save_the_plot: bool, save_dir):
        """Plot the actor losses"""
        plt.plot(range(len(total_actor_losses)), total_actor_losses)
        plt.xlabel('Training Steps')
        plt.ylabel('Total Actor Loss')
        plt.title('Total Actor Loss through steps')
        if save_the_plot: plt.savefig(os.path.join(save_dir, f"actor_losses_final_PPO.png"))
        plt.show()

    def plot_critic_losses(self, total_critic_losses: list, save_the_plot: bool, save_dir):
        plt.plot(range(len(total_critic_losses)), total_critic_losses)
        plt.xlabel('Training Steps')
        plt.ylabel('Total Critic Loss')
        plt.title('Total Critic Loss through steps')
        if save_the_plot: plt.savefig(os.path.join(save_dir, f"critic_losses_final_PPO.png"))
        plt.show()