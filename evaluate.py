import gymnasium
import highway_env
from matplotlib import pyplot as plt
import numpy as np
import torch
import random

import implementation_vpg as vpg

from implementations_dqn import DQN, DDQNAgent
from implementation_ppo_clip import ACNetwork, PPO

import os


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

env_name = "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')


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



dqn_type = ''
print("Which type of DQN would you like to evaluate? (1 for Double DQN, 2 for Dueling DQN, 3 for PPO, 4 for VPG)")
while dqn_type not in ['1', '2', '3', '4']:
    dqn_type = input("Enter your choice (1, 2, 3 or 4): ")
    if dqn_type not in ['1', '2', '3', '4']:
        print("Invalid choice. Please enter 1, 2, 3 or 4.")

dqn_name = "Double_DQN" if dqn_type == '1' else "Dueling_DQN" if dqn_type == '2' else "VPG" if dqn_type == '3' else "PPO" if dqn_type == '4' else "Unknown_DQN_Type"

dir = f"results_{dqn_name}"

if dqn_name == "Double_DQN":
    print("Evaluating Double DQN...")
    q_network_1 = DQN(state_dim, num_actions)
    target_network = DQN(state_dim, num_actions)

    q_network_1.load_state_dict(torch.load(os.path.join(dir, f'ddqn_agent_final_{dqn_name}.pth'))['q_network_1_state_dict'])
    target_network.load_state_dict(torch.load(os.path.join(dir, f'ddqn_agent_final_{dqn_name}.pth'))['target_network_state_dict'])

    # Initialize your model and load parameters
    agent = DDQNAgent(env=env,
                    q_network_1=q_network_1,
                    target_network=target_network,
                    discount_factor=0.8,
                    epsilon=0,   # no exploration during evaluation
                    buffer_size=10000,
                    batch_size=64)

elif dqn_name == "Dueling_DQN":
    print("Evaluating Dueling DQN...")
    q_network_1 = DQN(state_dim, num_actions, enable_dueling_dqn=True)
    target_network = DQN(state_dim, num_actions, enable_dueling_dqn=True)

    q_network_1.load_state_dict(torch.load(os.path.join(dir, f'ddqn_agent_final_{dqn_name}.pth'))['q_network_1_state_dict'])
    target_network.load_state_dict(torch.load(os.path.join(dir, f'ddqn_agent_final_{dqn_name}.pth'))['target_network_state_dict'])

    # Initialize your model and load parameters
    agent = DDQNAgent(env=env,
                    q_network_1=q_network_1,
                    target_network=target_network,
                    discount_factor=0.8,
                    epsilon=0,   # no exploration during evaluation
                    buffer_size=10000,
                    batch_size=64)

elif dqn_name == "VPG":
    print("Evaluating VPG...")

    network = vpg.Network(state_dim, num_actions)
    network.load_state_dict(torch.load(os.path.join(dir, f'drl_agent_final_{dqn_name}.pth')))

    agent = vpg.Vanilla_Policy_Gradient(network)

elif dqn_name == "PPO":
    print("Evaluating PPO...")

    dropout = 0.3  # Example dropout rate, adjust as needed

    actor = ACNetwork(state_dim, num_actions)
    critic = ACNetwork(state_dim, 1)

    actor.load_state_dict(torch.load(os.path.join(dir, f'drl_agent_final_{dqn_name}.pth'))['actor_state_dict'])
    critic.load_state_dict(torch.load(os.path.join(dir, f'drl_agent_final_{dqn_name}.pth'))['critic_state_dict'])

    agent = PPO(env, actor=actor, critic=critic)

else:
    raise NotImplementedError("DQN type not implemented yet.")



# Evaluation loop
state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

episode_returns = []    # Used to store the return after each trajectory and plot them

while episode <= 10:
    episode_steps += 1

    # Select the action to be performed by the agent
    if isinstance(agent, DDQNAgent):
        action = agent.select_action(state)
    elif isinstance(agent, vpg.Vanilla_Policy_Gradient):
        policy = agent.get_policy(agent.model, state)
        action, log_prob_action = agent.select_action(policy)
    else:
        policy = agent.get_policy(state)

        # Select the action to be performed by the agent
        action, log_prob = agent.select_action(policy)

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    state, reward, done, truncated, _ = env.step(action)

    state = state.reshape(-1)
    env.render()

    episode_return += float(reward)

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        episode_returns.append(episode_return)

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()


# Plot returns
plt.plot(range(len(episode_returns)), episode_returns)
plt.xlabel('Episode')
plt.ylabel('Episode Return')
plt.title('Episode Returns through episodes')
plt.show()