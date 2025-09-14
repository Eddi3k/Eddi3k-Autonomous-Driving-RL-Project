import gymnasium
import numpy as np
import torch
from torch import nn
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import collections
from typing import Tuple, Optional
import os

NUMBER_HIDDEN_FEATURES = 256

class DQN(nn.Module):

    def __init__(self, state_dim: int, num_actions: int, enable_dueling_dqn: bool = False):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn
        self.fc1 = nn.Linear(state_dim, NUMBER_HIDDEN_FEATURES)

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(NUMBER_HIDDEN_FEATURES, NUMBER_HIDDEN_FEATURES)
            self.value = nn.Linear(NUMBER_HIDDEN_FEATURES, 1)

            self.fc_advantages = nn.Linear(NUMBER_HIDDEN_FEATURES, NUMBER_HIDDEN_FEATURES)
            self.advantages = nn.Linear(NUMBER_HIDDEN_FEATURES, num_actions)

        else:
            self.fc2 = nn.Linear(NUMBER_HIDDEN_FEATURES, NUMBER_HIDDEN_FEATURES)
            self.output = nn.Linear(NUMBER_HIDDEN_FEATURES, num_actions)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value computation
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            #Advantages computation
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Compute Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            x = F.relu(self.fc2(x))
            Q = self.output(x)

        return Q
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


# ---------------------------------------- An Experience Replay Buffer is used to stabilize training ----------------------------------------

# Define the Experience namedtuple to store transitions
field_names = ['state', 'action', 'reward', 'next_state', 'done']
Experience = collections.namedtuple('Experience', field_names=field_names)

class ExperienceReplayBuffer:
    '''A simple fixed-size experience replay buffer to store transitions for training the agent'''

    def __init__(self,
                 buffer_size: int,
                 batch_size: int):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=buffer_size)


# --------------------------------------------- Define the Double DQN / Dueling DQN agent class ---------------------------------------------

class DDQNAgent:
    '''A Double DQN agent that uses an experience replay buffer to train the Q-network'''

    def __init__(self,
                 env: gymnasium.Env,
                 q_network_1: nn.Module,
                 target_network: nn.Module,
                 discount_factor,  # Discount factor for future rewards
                 buffer_size: int = 10000,    # TODO: check for the size of the buffer
                 batch_size: int = 128,
                 alpha = 1e-3,  # Learning rate for the Q-network
                 epsilon = 1.0,  # Initial exploration rate for the epsilon-greedy policy
                 epsilon_decay_rate = 0.995,  # Decay rate for the exploration rate
                 epsilon_min = 0.01,  # Minimum exploration rate
                 target_update_freq: int = 50):    # Frequency with which the target network is updated
        
        self.env = env

        self.q_network_1 = q_network_1
        self.target_network = target_network

        self.update_target_network()  # Initialize the target network with the same weights as the main network

        self.replay_buffer = ExperienceReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        self.alpha = alpha
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        
        self.target_update_freq = target_update_freq
        self.optimizer = torch.optim.Adam(self.q_network_1.parameters(), lr=self.alpha)

        self.step_count = 0     # Counter used to keep track of the number of times the agent has updated the Q-network


    # ----------------------------------- Select actions and update the Q-network -----------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using the epsilon-greedy policy"""

        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network_1(state_tensor)
            return q_values.argmax().item()  # Return the index of the action with the highest Q-value


    def update_q_network(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor):
        '''Update the Q-network using the Double DQN algorithm'''

        # Current Q-values for taken actions
        current_q_values = self.q_network_1(states).gather(1, actions.unsqueeze(1))

        # Double DQN: Use the main network to select actions, target network to evaluate them
        with torch.no_grad():
            # Select best actions using main network
            next_actions = self.q_network_1(next_states).argmax(1, keepdim=True)
            # Evaluate the selected actions using the target network
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Compute target Q-values
            target_q_values = rewards.unsqueeze(1) + self.discount_factor * next_q_values * (1 - dones.unsqueeze(1))

        # Compute the loss and update
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network_1.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Debug: Check for NaN
        if torch.isnan(loss):
            print("NaN loss detected!")
            return float('inf')     # If the loss is NaN, return a high value

        if loss.item() > 300:
            print(f"High loss detected!")

        return loss.item()
    

    def update_target_network(self):
        '''Copy weights from main network to target network'''
        self.target_network.load_state_dict(self.q_network_1.state_dict())
        print("Target network updated.")


    def epsilon_decay(self):
        '''Decay the exploration rate'''
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)


    # ----------------------------------- Store transitions and sample experiences -----------------------------------

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        '''Store a transition in the replay buffer'''
        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.replay_buffer.buffer.append(experience)

    def sample_experience(self) -> Optional[Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]:
        '''Sample a batch of experiences from the replay buffer'''
        if len(self.replay_buffer.buffer) < self.replay_buffer.batch_size:
            # Not enough experiences in the buffer to sample a batch
            return None
        sample_batch = random.sample(self.replay_buffer.buffer, self.replay_buffer.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample_batch)

        # Convert to numpy arrays first, then to tensors
        return (torch.FloatTensor(np.array(states)), 
                torch.LongTensor(np.array(actions)), 
                torch.FloatTensor(np.array(rewards)), 
                torch.FloatTensor(np.array(next_states)), 
                torch.FloatTensor(np.array(dones)))


    # ----------------------------------- Train the agent -----------------------------------
    def train(self):
        """Train the agent using a sampled batch of experiences"""

        experience_batch = self.sample_experience()

        if experience_batch is None:
            return None

        states, actions, rewards, next_states, dones = experience_batch
        
        # Update the Q-network
        loss = self.update_q_network(states, actions, rewards, next_states, dones)

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss
    

    # ------------------------ Save the agent's parameters to a file ------------------------
    def save(self, file_path: str):
        """Save the agent's parameters to a file"""
        torch.save({
            'q_network_1_state_dict': self.q_network_1.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, file_path)
        print(f"Agent parameters saved to {file_path}")



    # -------------------------------- Plot results --------------------------------

    def plot_mean_losses(self, mean_losses: list, dqn_name, save_the_plot: bool, save_dir):
        '''Plot the mean losses'''
        plt.plot(range(len(mean_losses)), mean_losses)
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Loss')
        plt.title('Mean Loss through steps')
        if save_the_plot: plt.savefig(os.path.join(save_dir, f"mean_losses_final_{dqn_name}.png"))
        plt.show()

    def plot_returns(self, episode_returns: list, dqn_name, save_the_plot: bool, save_dir):
        '''Plot the episode returns'''
        mean_return = np.mean(episode_returns)

        plt.plot(range(len(episode_returns)), episode_returns)
        plt.xlabel('Episode')
        plt.ylabel('Episode Return')
        plt.title('Episode Returns through episodes')
        if save_the_plot: plt.savefig(os.path.join(save_dir, f"episode_returns_final_{dqn_name}.png"))
        plt.show()

    def plot_mean_returns(self, mean_returns: list, dqn_name, save_the_plot: bool, save_dir):
        """Plot the average episode returns, taken considering windows of  "WINDOW_SIZE" steps"""
        plt.plot(range(len(mean_returns)), mean_returns)
        plt.xlabel('Episode')
        plt.ylabel('Average Episode Return')
        plt.title('Average Episode Returns through episodes')
        if save_the_plot: plt.savefig(os.path.join(save_dir, f"mean_episode_returns_final_{dqn_name}.png"))
        plt.show()