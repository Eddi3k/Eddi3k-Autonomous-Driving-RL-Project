import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import matplotlib.pyplot as plt

NUMBER_HIDDEN_FEATURES = 256
LEARNING_RATE = 1e-3

class Network(nn.Module):
    def __init__(self, number_observation_features: int, number_actions: int):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(number_observation_features, NUMBER_HIDDEN_FEATURES)
        self.fc2 = nn.Linear(NUMBER_HIDDEN_FEATURES, NUMBER_HIDDEN_FEATURES)
        self.fc3 = nn.Linear(NUMBER_HIDDEN_FEATURES, number_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Vanilla_Policy_Gradient(nn.Module):
    def __init__(self, model: Network):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)


    def get_policy(self, model: Network, observation: np.ndarray) -> distributions.Categorical:
        '''Returns the policy distribution for a given observation.

        Args:
            model (Network): The policy network.
            observation (np.ndarray): The observation (== state) for which to compute the policy.

        Returns:
            distributions.Categorical: Multinomial distribution parameterized by model logits.
        '''

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        logits = model(observation_tensor)

        # Categorical will also normalize the logits for us
        return distributions.Categorical(logits=logits)
    

    def select_action(self, policy: distributions.Categorical) -> tuple[int, torch.Tensor]:
        '''Samples an action from the given policy distribution.

        Returns:
            tuple[int, float]: action and its probability
        '''

        action = policy.sample()

        # We keep track of log probabilities instead of raw action probabilities as it makes gradient ascent easier behind the scenes
        log_probability_action = policy.log_prob(action)

        return int(action.item()), log_probability_action


    def calculate_loss(self, epoch_log_probabity_actions: torch.Tensor, epoch_action_rewards: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the "loss" required to get the policy gradient.\n
        WARNING: this is NOT the real loss, as it is just the sum of the
        log probabilities of the actions taken, weighted by their rewards.
        The real loss is computed after the back-propagation step.

        Args:
            epoch_log_probabity_actions (torch.Tensor): Log probabilities of actions taken during the epoch.
            epoch_action_rewards (torch.Tensor): Rewards received for actions taken during the epoch.

        Returns:
            float: The calculated pseudo-loss value.
        '''

        # Policy gradient: maximize E[log π(a|s) * R]
        # We minimize the negative, so return -E[log π(a|s) * R]
        return -(epoch_log_probabity_actions * epoch_action_rewards).mean()
    

    def save(self, path: str) -> None:
        '''Save the model parameters to a file.'''
        torch.save(self.model.state_dict(), path)


    def plot_returns(self, episode_returns: list[float], save_the_plot: bool, save_dir: str):
        '''Plot the episode returns.'''
        plt.plot(range(len(episode_returns)), episode_returns)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Episode Returns through episodes")
        if save_the_plot: plt.savefig(os.path.join(save_dir, "episode_returns_final_VPG.png"))
        plt.show()